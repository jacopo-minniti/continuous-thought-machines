import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

from models.modules import ParityBackbone, SynapseUNET, Squeeze, SuperLinear, LearnableFourierPositionalEncoding, MultiLearnableFourierPositionalEncoding, CustomRotationalEmbedding, CustomRotationalEmbedding1D, ShallowWide
from models.resnet import prepare_resnet_backbone
from models.utils import compute_normalized_entropy

from models.constants import (
    VALID_NEURON_SELECT_TYPES,
    VALID_BACKBONE_TYPES,
    VALID_POSITIONAL_EMBEDDING_TYPES
)

class ContinuousThoughtMachine(nn.Module, PyTorchModelHubMixin):
    """
    Continuous Thought Machine (CTM).

    Technical report: https://arxiv.org/abs/2505.05522

    Interactive Website: https://pub.sakana.ai/ctm/

    Blog: https://sakana.ai/ctm/

    Thought takes time and reasoning is a process. 
    
    The CTM consists of three main ideas:
    1. The use of internal recurrence, enabling a dimension over which a concept analogous to thought can occur. 
    1. Neuron-level models, that compute post-activations by applying private (i.e., on a per-neuron basis) MLP 
       models to a history of incoming pre-activations.
    2. Synchronisation as representation, where the neural activity over time is tracked and used to compute how 
       pairs of neurons synchronise with one another over time. This measure of synchronisation is the representation 
       with which the CTM takes action and makes predictions.


    Args:
        iterations (int): Number of internal 'thought' ticks (T, in paper).
        d_model (int): Core dimensionality of the CTM's latent space (D, in paper).
                       NOTE: Note that this is NOT the representation used for action or prediction, but rather that which
                       is fully internal to the model and not directly connected to data.
        d_input (int): Dimensionality of projected attention outputs or direct input features.
        heads (int): Number of attention heads.
        n_synch_out (int): Number of neurons used for output synchronisation (D_out, in paper).
        n_synch_action (int): Number of neurons used for action/attention synchronisation (D_action, in paper).
        synapse_depth (int): Depth of the synapse model (U-Net if > 1, else MLP).
        memory_length (int): History length for Neuron-Level Models (M, in paper).
        deep_nlms (bool): Use deeper (2-layer) NLMs if True, else linear.
                        NOTE: we almost always use deep NLMs, but a linear NLM is faster.
        memory_hidden_dims (int): Hidden dimension size for deep NLMs.
        do_layernorm_nlm (bool): Apply LayerNorm within NLMs.
                        NOTE: we never set this to true in the paper. If you set this to true you will get strange behaviour,
                        but you can potentially encourage more periodic behaviour in the dynamics. Untested; be careful.
        backbone_type (str): Type of feature extraction backbone (e.g., 'resnet18-2', 'none').
        positional_embedding_type (str): Type of positional embedding for backbone features.
        out_dims (int): Output dimension size.
                        NOTE: projected from synchronisation!
        prediction_reshaper (list): Shape for reshaping predictions before certainty calculation (task-specific).
                        NOTE: this is used to compute certainty and is needed when applying softmax for probabilities
        dropout (float): Dropout rate.
        neuron_select_type (str): Neuron selection strategy ('first-last', 'random', 'random-pairing').
                        NOTE: some of this is legacy from our experimentation, but all three strategies are valid and useful. 
                            We dilineate exactly which strategies we use per experiment in the paper. 
                        - first-last: build a 'dense' sync matrix for output from the first D_out neurons and action from the 
                                      last D_action neurons. Flatten this matrix into the synchronisation representation. 
                                      This approach shares relationships for neurons and bottlenecks the gradients through them.
                                      NOTE: the synchronisation size will be (D_out/action * (D_out/action + 1))/2
                        - random: randomly select D_out neurons for the 'i' side pairings, and also D_out for the 'j' side pairings,
                                      also pairing those accross densely, resulting in a bottleneck roughly 2x as wide.
                                      NOTE: the synchronisation size will be (D_out/action * (D_out/action + 1))/2
                        - random-pairing (DEFAULT!): randomly select D_out neurons and pair these with another D_out neurons. 
                                      This results in much less bottlenecking and is the most up-to-date variant.
                                      NOTE: the synchronisation size will be D_out in this case; better control. 
        n_random_pairing_self (int): Number of neurons to select for self-to-self synch when random-pairing is used.
                        NOTE: when using random-pairing, i-to-i (self) synchronisation is rare, meaning that 'recovering a
                        snapshot representation' (see paper) is difficult. This alleviates that. 
                        NOTE: works fine when set to 0.
    """                               

    def __init__(self,
                 iterations,
                 d_model,
                 d_input,
                 heads,
                 n_synch_out,
                 n_synch_action,
                 synapse_depth,
                 memory_length,
                 deep_nlms,
                 memory_hidden_dims,
                 do_layernorm_nlm,
                 backbone_type,
                 positional_embedding_type,
                 out_dims,
                 prediction_reshaper=[-1],
                 dropout=0,
                 dropout_nlm=None,
                 neuron_select_type='random-pairing',  
                 n_random_pairing_self=0,
                 gamma=0.25,
                 probe_every=4,
                 probe_frac=0.25,
                 cf_projection_dim=128,
                 ):
        super(ContinuousThoughtMachine, self).__init__()

        # --- Core Parameters ---
        self.iterations = iterations
        self.d_model = d_model
        self.d_input = d_input
        self.memory_length = memory_length
        self.prediction_reshaper = prediction_reshaper
        self.n_synch_out = n_synch_out
        self.n_synch_action = n_synch_action
        self.backbone_type = backbone_type
        self.out_dims = out_dims
        self.positional_embedding_type = positional_embedding_type
        self.neuron_select_type = neuron_select_type
        self.memory_length = memory_length
        dropout_nlm = dropout if dropout_nlm is None else dropout_nlm
        self.gamma = float(gamma)
        self.probe_every = int(probe_every) if probe_every is not None else 0
        self.probe_frac = 0.0 if probe_frac is None else float(np.clip(probe_frac, 0.0, 1.0))
        self.cf_projection_dim_target = cf_projection_dim
        self.latest_gate_loss = None
        self.latest_gate_metrics = {}
        self.latest_gate_sequence = None

        # --- Assertions ---
        self.verify_args()

        # --- Input Processing  ---
        d_backbone = self.get_d_backbone()
        self.set_initial_rgb()
        self.set_backbone()
        self.positional_embedding = self.get_positional_embedding(d_backbone)
        self.kv_proj = nn.Sequential(nn.LazyLinear(self.d_input), nn.LayerNorm(self.d_input)) if heads else None
        self.q_proj = nn.LazyLinear(self.d_input) if heads else None
        self.attention = nn.MultiheadAttention(self.d_input, heads, dropout, batch_first=True) if heads else None
        self.attention_readout = nn.Linear(self.d_input, self.d_model)
        
        # --- Core CTM Modules ---
        self.synapses = self.get_synapses(synapse_depth, d_model, dropout)
        self.trace_processor = self.get_neuron_level_models(deep_nlms, do_layernorm_nlm, memory_length, memory_hidden_dims, d_model, dropout_nlm)

        #  --- Start States ---
        self.register_parameter('start_activated_state', nn.Parameter(torch.zeros((d_model)).uniform_(-math.sqrt(1/(d_model)), math.sqrt(1/(d_model)))))
        self.register_parameter('start_trace', nn.Parameter(torch.zeros((d_model, memory_length)).uniform_(-math.sqrt(1/(d_model+memory_length)), math.sqrt(1/(d_model+memory_length)))))

        # --- Synchronisation ---
        self.neuron_select_type_out, self.neuron_select_type_action = self.get_neuron_select_type()
        self.synch_representation_size_action = self.calculate_synch_representation_size(self.n_synch_action)
        self.synch_representation_size_out = self.calculate_synch_representation_size(self.n_synch_out)
        
        for synch_type, size in (('action', self.synch_representation_size_action), ('out', self.synch_representation_size_out)):
            print(f"Synch representation size {synch_type}: {size}")
        if self.synch_representation_size_action:  # if not zero
            self.set_synchronisation_parameters('action', self.n_synch_action, n_random_pairing_self)
        self.set_synchronisation_parameters('out', self.n_synch_out, n_random_pairing_self)

        if self.synch_representation_size_action:
            gate_hidden = max(64, min(256, self.synch_representation_size_action * 2))
            self.gate_head = nn.Sequential(
                nn.LayerNorm(self.synch_representation_size_action),
                nn.Linear(self.synch_representation_size_action, gate_hidden),
                nn.SiLU(),
                nn.Linear(gate_hidden, 1)
            )
        else:
            self.gate_head = None

        cf_dim = min(self.cf_projection_dim_target, self.synch_representation_size_out) if self.synch_representation_size_out else 0
        self.cf_projection_dim = cf_dim
        self.cf_projector = nn.Linear(self.synch_representation_size_out, cf_dim, bias=False) if cf_dim else None

        # --- Output Procesing ---
        self.output_projector = nn.Sequential(nn.LazyLinear(self.out_dims))

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision=None,
        cache_dir=None,
        force_download=False,
        proxies=None,
        resume_download=None,
        local_files_only=False,
        token=None,
        map_location="cpu",
        strict=False,
        **model_kwargs,
    ):
        """Override to handle lazy weights initialization."""
        model = cls(**model_kwargs).to(map_location)

        # The CTM contains Lazy modules, so we must run a dummy forward pass to initialize them
        if "imagenet" in model_id:
            dummy_input = torch.randn(1, 3, 224, 224, device=map_location)
        elif "maze-large" in model_id:
            dummy_input = torch.randn(1, 3, 99, 99, device=map_location)
        else:
            raise NotImplementedError

        with torch.no_grad():
            _ = model(dummy_input)

        model_file = hf_hub_download(
            repo_id=model_id,
            filename="model.safetensors",
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            token=token,
            local_files_only=local_files_only,
        )
        from safetensors.torch import load_model as load_model_as_safetensor
        load_model_as_safetensor(model, model_file, strict=strict, device=map_location)

        model.eval()
        return model

    # --- Core CTM Methods ---

    def compute_synchronisation(self, activated_state, decay_alpha, decay_beta, r, synch_type):
        """
        Computes synchronisation to be used as a vector representation. 

        A neuron has what we call a 'trace', which is a history (time series) that changes with internal
        recurrence. i.e., it gets longer with every internal tick. There are pre-activation traces
        that are used in the NLMs and post-activation traces that, in theory, are used in this method. 

        We define sychronisation between neuron i and j as the dot product between their respective
        time series. Since there can be many internal ticks, this process can be quite compute heavy as it
        involves many dot products that repeat computation at each step.
        
        Therefore, in practice, we update the synchronisation based on the current post-activations,
        which we call the 'activated state' here. This is possible because the inputs to synchronisation 
        are only updated recurrently at each step, meaning that there is a linear recurrence we can
        leverage. 
        
        See Appendix TODO of the Technical Report (TODO:LINK) for the maths that enables this method.
        """

        if synch_type == 'action': # Get action parameters
            n_synch = self.n_synch_action
            neuron_indices_left = self.action_neuron_indices_left
            neuron_indices_right = self.action_neuron_indices_right
        elif synch_type == 'out': # Get input parameters
            n_synch = self.n_synch_out
            neuron_indices_left = self.out_neuron_indices_left
            neuron_indices_right = self.out_neuron_indices_right
        
        if self.neuron_select_type in ('first-last', 'random'):
            # For first-last and random, we compute the pairwise sync between all selected neurons
            if self.neuron_select_type == 'first-last':
                if synch_type == 'action': # Use last n_synch neurons for action
                    selected_left = selected_right = activated_state[:, -n_synch:]
                elif synch_type == 'out': # Use first n_synch neurons for out
                    selected_left = selected_right = activated_state[:, :n_synch]
            else: # Use the randomly selected neurons
                selected_left = activated_state[:, neuron_indices_left]
                selected_right = activated_state[:, neuron_indices_right]
            
            # Compute outer product of selected neurons
            outer = selected_left.unsqueeze(2) * selected_right.unsqueeze(1)
            # Resulting matrix is symmetric, so we only need the upper triangle
            i, j = torch.triu_indices(n_synch, n_synch)
            pairwise_product = outer[:, i, j]
            
        elif self.neuron_select_type == 'random-pairing':
            # For random-pairing, we compute the sync between specific pairs of neurons
            left = activated_state[:, neuron_indices_left]
            right = activated_state[:, neuron_indices_right]
            pairwise_product = left * right
        else:
            raise ValueError("Invalid neuron selection type")
        
        
        
        # Compute synchronisation recurrently
        if decay_alpha is None or decay_beta is None:
            decay_alpha = pairwise_product
            decay_beta = torch.ones_like(pairwise_product)
        else:
            decay_alpha = r * decay_alpha + pairwise_product
            decay_beta = r * decay_beta + 1
        
        synchronisation = decay_alpha / (torch.sqrt(decay_beta))
        return synchronisation, decay_alpha, decay_beta

    def compute_features(self, x):
        """
        Compute the key-value features from the input data using the backbone. 
        """
        initial_rgb = self.initial_rgb(x)
        self.kv_features = self.backbone(initial_rgb)
        pos_emb = self.positional_embedding(self.kv_features)
        combined_features = (self.kv_features + pos_emb).flatten(2).transpose(1, 2)
        kv = self.kv_proj(combined_features)
        return kv

    def compute_certainty(self, current_prediction):
        """
        Compute the certainty of the current prediction.
        
        We define certainty as being 1-normalised entropy.

        For legacy reasons we stack that in a 2D vector as this can be used for optimisation later.
        """
        B = current_prediction.size(0)
        reshaped_pred = current_prediction.reshape([B] + self.prediction_reshaper)
        ne = compute_normalized_entropy(reshaped_pred)
        current_certainty = torch.stack((ne, 1-ne), -1)
        return current_certainty

    # --- Setup Methods ---

    def set_initial_rgb(self):
        """
        This is largely to accommodate training on grayscale images and is legacy, but it
        doesn't hurt the model in any way that we can tell.
        """
        if 'resnet' in self.backbone_type:
            self.initial_rgb = nn.LazyConv2d(3, 1, 1) # Adapts input channels lazily
        else:
            self.initial_rgb = nn.Identity()

    def get_d_backbone(self):
        """
        Get the dimensionality of the backbone output, to be used for positional embedding setup.

        This is a little bit complicated for resnets, but the logic should be easy enough to read below.        
        """
        if self.backbone_type == 'shallow-wide':
            return 2048
        elif self.backbone_type == 'parity_backbone':
            return self.d_input
        elif 'resnet' in self.backbone_type:
            if '18' in self.backbone_type or '34' in self.backbone_type: 
                if self.backbone_type.split('-')[1]=='1': return 64
                elif self.backbone_type.split('-')[1]=='2': return 128
                elif self.backbone_type.split('-')[1]=='3': return 256
                elif self.backbone_type.split('-')[1]=='4': return 512
                else:
                    raise NotImplementedError
            else:
                if self.backbone_type.split('-')[1]=='1': return 256
                elif self.backbone_type.split('-')[1]=='2': return 512
                elif self.backbone_type.split('-')[1]=='3': return 1024
                elif self.backbone_type.split('-')[1]=='4': return 2048
                else:
                    raise NotImplementedError
        elif self.backbone_type == 'none':
            return None
        else:
            raise ValueError(f"Invalid backbone_type: {self.backbone_type}")

    def set_backbone(self):
        """
        Set the backbone module based on the specified type.
        """
        if self.backbone_type == 'shallow-wide':
            self.backbone = ShallowWide()
        elif self.backbone_type == 'parity_backbone':
            d_backbone = self.get_d_backbone()
            self.backbone = ParityBackbone(n_embeddings=2, d_embedding=d_backbone)
        elif 'resnet' in self.backbone_type:
            self.backbone = prepare_resnet_backbone(self.backbone_type)
        elif self.backbone_type == 'none':
            self.backbone = nn.Identity()
        else:
            raise ValueError(f"Invalid backbone_type: {self.backbone_type}")

    def get_positional_embedding(self, d_backbone):
        """
        Get the positional embedding module.

        For Imagenet and mazes we used NO positional embedding, and largely don't think
        that it is necessary as the CTM can build up its own internal world model when
        observing.

        LearnableFourierPositionalEncoding:
            Implements Algorithm 1 from "Learnable Fourier Features for Multi-Dimensional
            Spatial Positional Encoding" (https://arxiv.org/pdf/2106.02795.pdf).
            Provides positional information for 2D feature maps.      

            (MultiLearnableFourierPositionalEncoding uses multiple feature scales)

        CustomRotationalEmbedding:
            Simple sinusoidal embedding to encourage interpretability
        """
        if self.positional_embedding_type == 'learnable-fourier':
            return LearnableFourierPositionalEncoding(d_backbone, gamma=1 / 2.5)
        elif self.positional_embedding_type == 'multi-learnable-fourier':
            return MultiLearnableFourierPositionalEncoding(d_backbone)
        elif self.positional_embedding_type == 'custom-rotational':
            return CustomRotationalEmbedding(d_backbone)
        elif self.positional_embedding_type == 'custom-rotational-1d':
            return CustomRotationalEmbedding1D(d_backbone)
        elif self.positional_embedding_type == 'none':
            return lambda x: 0  # Default no-op
        else:
            raise ValueError(f"Invalid positional_embedding_type: {self.positional_embedding_type}")

    def get_neuron_level_models(self, deep_nlms, do_layernorm_nlm, memory_length, memory_hidden_dims, d_model, dropout):
        """
        Neuron level models are one of the core innovations of the CTM. They apply separate MLPs/linears to 
        each neuron.
        NOTE: the name 'SuperLinear' is largely legacy, but its purpose is to apply separate linear layers
            per neuron. It is sort of a 'grouped linear' function, where the group size is equal to 1. 
            One could make the group size bigger and use fewer parameters, but that is future work.

        NOTE: We used GLU() nonlinearities because they worked well in practice. 
        """
        if deep_nlms:
            return nn.Sequential(
                nn.Sequential(
                    SuperLinear(in_dims=memory_length, out_dims=2 * memory_hidden_dims, N=d_model,
                                do_norm=do_layernorm_nlm, dropout=dropout),
                    nn.GLU(),
                    SuperLinear(in_dims=memory_hidden_dims, out_dims=2, N=d_model,
                                do_norm=do_layernorm_nlm, dropout=dropout),
                    nn.GLU(),
                    Squeeze(-1)
                )
            )
        else:
            return nn.Sequential(
                nn.Sequential(
                    SuperLinear(in_dims=memory_length, out_dims=2, N=d_model,
                                do_norm=do_layernorm_nlm, dropout=dropout),
                    nn.GLU(),
                    Squeeze(-1)
                )
            )

    def get_synapses(self, synapse_depth, d_model, dropout):
        """
        The synapse model is the recurrent model in the CTM. It's purpose is to share information
        across neurons. If using depth of 1, this is just a simple single layer with nonlinearity and layernomr.
        For deeper synapse models we use a U-NET structure with many skip connections. In practice this performs
        better as it enables multi-level information mixing.

        The intuition with having a deep UNET model for synapses is that the action of synaptic connections is
        not necessarily a linear one, and that approximate a synapose 'update' step in the brain is non trivial. 
        Hence, we set it up so that the CTM can learn some complex internal rule instead of trying to approximate
        it ourselves.
        """
        if synapse_depth == 1:
            return nn.Sequential(
                nn.Dropout(dropout),
                nn.LazyLinear(d_model * 2),
                nn.GLU(),
                nn.LayerNorm(d_model)
            )
        else:
            return SynapseUNET(d_model, synapse_depth, 16, dropout)  # hard-coded minimum width of 16; future work TODO.

    def set_synchronisation_parameters(self, synch_type: str, n_synch: int, n_random_pairing_self: int = 0):
            """
            1. Set the buffers for selecting neurons so that these indices are saved into the model state_dict.
            2. Set the parameters for learnable exponential decay when computing synchronisation between all 
                neurons.
            """
            assert synch_type in ('out', 'action'), f"Invalid synch_type: {synch_type}"
            left, right = self.initialize_left_right_neurons(synch_type, self.d_model, n_synch, n_random_pairing_self)
            synch_representation_size = self.synch_representation_size_action if synch_type == 'action' else self.synch_representation_size_out
            self.register_buffer(f'{synch_type}_neuron_indices_left', left)
            self.register_buffer(f'{synch_type}_neuron_indices_right', right)
            self.register_parameter(f'decay_params_{synch_type}', nn.Parameter(torch.zeros(synch_representation_size), requires_grad=True))

    def initialize_left_right_neurons(self, synch_type, d_model, n_synch, n_random_pairing_self=0):
        """
        Initialize the left and right neuron indices based on the neuron selection type.
        This complexity is owing to legacy experiments, but we retain that these types of
        neuron selections are interesting to experiment with.
        """
        if self.neuron_select_type=='first-last':
            if synch_type == 'out':
                neuron_indices_left = neuron_indices_right = torch.arange(0, n_synch)
            elif synch_type == 'action':
                neuron_indices_left = neuron_indices_right = torch.arange(d_model-n_synch, d_model)

        elif self.neuron_select_type=='random':
            neuron_indices_left = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))
            neuron_indices_right = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))

        elif self.neuron_select_type=='random-pairing':
            assert n_synch > n_random_pairing_self, f"Need at least {n_random_pairing_self} pairs for {self.neuron_select_type}"
            neuron_indices_left = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))
            neuron_indices_right = torch.concatenate((neuron_indices_left[:n_random_pairing_self], torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch-n_random_pairing_self))))

        device = self.start_activated_state.device
        return neuron_indices_left.to(device), neuron_indices_right.to(device)

    def get_neuron_select_type(self):
        """
        Another helper method to accomodate our legacy neuron selection types. 
        TODO: additional experimentation and possible removal of 'first-last' and 'random'
        """
        print(f"Using neuron select type: {self.neuron_select_type}")
        if self.neuron_select_type == 'first-last':
            neuron_select_type_out, neuron_select_type_action = 'first', 'last'
        elif self.neuron_select_type in ('random', 'random-pairing'):
            neuron_select_type_out = neuron_select_type_action = self.neuron_select_type
        else:
            raise ValueError(f"Invalid neuron selection type: {self.neuron_select_type}")
        return neuron_select_type_out, neuron_select_type_action

    # --- Utilty Methods ---

    def verify_args(self):
        """
        Verify the validity of the input arguments to ensure consistent behaviour. 
        Specifically when selecting neurons for sychronisation using 'first-last' or 'random',
        one needs the right number of neurons
        """
        assert self.neuron_select_type in VALID_NEURON_SELECT_TYPES, \
            f"Invalid neuron selection type: {self.neuron_select_type}"
        
        assert self.backbone_type in VALID_BACKBONE_TYPES + ['none'], \
            f"Invalid backbone_type: {self.backbone_type}"
        
        assert self.positional_embedding_type in VALID_POSITIONAL_EMBEDDING_TYPES + ['none'], \
            f"Invalid positional_embedding_type: {self.positional_embedding_type}"
        
        if self.neuron_select_type == 'first-last':
            assert self.d_model >= (self.n_synch_out + self.n_synch_action), \
                "d_model must be >= n_synch_out + n_synch_action for neuron subsets"

        if self.backbone_type=='none' and self.positional_embedding_type!='none':
            raise AssertionError("There should be no positional embedding if there is no backbone.")

    def calculate_synch_representation_size(self, n_synch):
        """
        Calculate the size of the synchronisation representation based on neuron selection type.
        """
        if self.neuron_select_type == 'random-pairing':
            synch_representation_size = n_synch
        elif self.neuron_select_type in ('first-last', 'random'):
            synch_representation_size = (n_synch * (n_synch + 1)) // 2
        else:
            raise ValueError(f"Invalid neuron selection type: {self.neuron_select_type}")
        return synch_representation_size

    def _project_counterfactual(self, synchronisation_vector):
        if self.cf_projector is None or synchronisation_vector is None:
            return None
        return self.cf_projector(synchronisation_vector)

    def _sample_probe_indices(self, batch_size, device):
        if self.probe_frac <= 0 or self.probe_frac is None:
            return torch.empty(0, dtype=torch.long, device=device)
        if self.probe_frac >= 1.0:
            return torch.arange(batch_size, device=device)
        mask = torch.rand(batch_size, device=device) < self.probe_frac
        return mask.nonzero(as_tuple=False).flatten()

    def _compute_gate_target(
        self,
        prev_activated_state,
        o_t,
        state_trace_prev,
        idx,
        decay_alpha_out_prev,
        decay_beta_out_prev,
        r_out,
        synch_out_prev,
        device,
    ):
        if self.cf_projector is None or synch_out_prev is None or idx.numel() == 0:
            return None, None

        with torch.no_grad():
            v_t = self._project_counterfactual(synch_out_prev[idx])
            if v_t is None:
                return None, None

            ingest_state = self.synapses(o_t[idx])
            reflect_state = self.synapses(prev_activated_state[idx])

            ingest_trace = torch.cat((state_trace_prev[idx, :, 1:], ingest_state.unsqueeze(-1)), dim=-1)
            reflect_trace = torch.cat((state_trace_prev[idx, :, 1:], reflect_state.unsqueeze(-1)), dim=-1)

            ingest_activation = self.trace_processor(ingest_trace)
            reflect_activation = self.trace_processor(reflect_trace)

            da_prev = None if decay_alpha_out_prev is None else decay_alpha_out_prev[idx]
            db_prev = None if decay_beta_out_prev is None else decay_beta_out_prev[idx]
            r_slice = r_out[idx]

            synch_ing, _, _ = self.compute_synchronisation(ingest_activation, da_prev, db_prev, r_slice, synch_type='out')
            synch_ref, _, _ = self.compute_synchronisation(reflect_activation, da_prev, db_prev, r_slice, synch_type='out')

            v_ing = self._project_counterfactual(synch_ing)
            v_ref = self._project_counterfactual(synch_ref)
            if v_ing is None or v_ref is None:
                return None, None

            delta_ing = torch.linalg.norm(v_ing - v_t, dim=-1)
            delta_ref = torch.linalg.norm(v_ref - v_t, dim=-1)
            r_star = torch.where(delta_ing > delta_ref, torch.zeros_like(delta_ing), torch.ones_like(delta_ref)).unsqueeze(-1)
            open_frac = (r_star == 0).float().mean().item()

        return r_star.to(device), open_frac

    def get_gate_loss(self):
        return self.latest_gate_loss

    def get_gate_metrics(self):
        return self.latest_gate_metrics

    def get_latest_gate_sequence(self):
        return self.latest_gate_sequence




    def forward(self, x, track=False):
        B = x.size(0)
        device = x.device

        self.latest_gate_loss = None
        self.latest_gate_metrics = {}
        self.latest_gate_sequence = None
        gate_losses = []
        gate_value_sum = 0.0
        gate_value_count = 0
        gate_probe_accuracy_sum = 0.0
        gate_probe_open_sum = 0.0
        gate_probe_count = 0
        gate_history = []

        gate_supervision_active = (
            self.training
            and self.gamma > 0
            and self.probe_every > 0
            and self.probe_frac > 0
            and self.gate_head is not None
            and self.cf_projector is not None
        )

        # --- Tracking Initialization ---
        pre_activations_tracking = []
        post_activations_tracking = []
        synch_out_tracking = []
        synch_action_tracking = []
        attention_tracking = []

        # --- Featurise Input Data ---
        kv = self.compute_features(x)

        # --- Initialise Recurrent State ---
        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1) # Shape: (B, H, T)
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1) # Shape: (B, H)

        # --- Prepare Storage for Outputs per Iteration ---
        predictions = torch.empty(B, self.out_dims, self.iterations, device=device, dtype=torch.float32)
        certainties = torch.empty(B, 2, self.iterations, device=device, dtype=torch.float32)

        # --- Initialise Recurrent Synch Values  ---
        decay_alpha_action, decay_beta_action = None, None
        self.decay_params_action.data = torch.clamp(self.decay_params_action, 0, 15)  # Fix from github user: kuviki
        self.decay_params_out.data = torch.clamp(self.decay_params_out, 0, 15)
        r_action = torch.exp(-self.decay_params_action).unsqueeze(0).repeat(B, 1)
        r_out = torch.exp(-self.decay_params_out).unsqueeze(0).repeat(B, 1)

        synchronisation_out_prev, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, None, None, r_out, synch_type='out')
        synchronisation_out_prev = synchronisation_out_prev.detach()

        # --- Recurrent Loop  ---
        for stepi in range(self.iterations):
            state_trace_prev = state_trace
            prev_activated_state = activated_state

            # --- Calculate Synchronisation for Input Data Interaction ---
            synchronisation_action, decay_alpha_action, decay_beta_action = self.compute_synchronisation(prev_activated_state, decay_alpha_action, decay_beta_action, r_action, synch_type='action')

            # --- Interact with Data via Attention ---
            q = self.q_proj(synchronisation_action).unsqueeze(1)
            attn_out, attn_weights = self.attention(q, kv, kv, average_attn_weights=False, need_weights=True)
            attn_out = attn_out.squeeze(1)
            o_t = self.attention_readout(attn_out)

            if self.gate_head is not None:
                gate_logits = self.gate_head(synchronisation_action)
                gate_values = torch.sigmoid(gate_logits)
                gate_history.append(gate_values.detach())
                gate_value_sum += gate_values.detach().mean().item()
                gate_value_count += 1
            else:
                gate_logits = None
                gate_values = torch.ones(prev_activated_state.size(0), 1, device=device, dtype=prev_activated_state.dtype)

            # --- Apply Synapses ---
            synapse_input = gate_values * prev_activated_state + (1 - gate_values) * o_t
            state = self.synapses(synapse_input)
            state_trace = torch.cat((state_trace_prev[:, :, 1:], state.unsqueeze(-1)), dim=-1)

            # --- Apply Neuron-Level Models ---
            activated_state = self.trace_processor(state_trace)

            # --- Calculate Synchronisation for Output Predictions ---
            decay_alpha_out_prev = decay_alpha_out
            decay_beta_out_prev = decay_beta_out
            synchronisation_out, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type='out')

            # --- Training-time counterfactual supervision ---
            if gate_supervision_active and (stepi % self.probe_every == 0) and gate_logits is not None and synchronisation_out_prev is not None:
                probe_idx = self._sample_probe_indices(B, device)
                if probe_idx.numel() > 0:
                    r_star, open_frac = self._compute_gate_target(
                        prev_activated_state,
                        o_t,
                        state_trace_prev,
                        probe_idx,
                        decay_alpha_out_prev,
                        decay_beta_out_prev,
                        r_out,
                        synchronisation_out_prev,
                        device,
                    )
                    if r_star is not None:
                        gate_loss = F.binary_cross_entropy_with_logits(gate_logits[probe_idx], r_star)
                        gate_losses.append(gate_loss)
                        gate_probe_count += 1
                        gate_probe_open_sum += open_frac
                        gate_predictions = (torch.sigmoid(gate_logits[probe_idx].detach()) >= 0.5).float()
                        gate_probe_accuracy_sum += (gate_predictions == r_star).float().mean().item()

            synchronisation_out_prev = synchronisation_out.detach()

            # --- Get Predictions and Certainties ---
            current_prediction = self.output_projector(synchronisation_out)
            current_certainty = self.compute_certainty(current_prediction)

            predictions[..., stepi] = current_prediction
            certainties[..., stepi] = current_certainty

            # --- Tracking ---
            if track:
                pre_activations_tracking.append(state_trace[:,:,-1].detach().cpu().numpy())
                post_activations_tracking.append(activated_state.detach().cpu().numpy())
                attention_tracking.append(attn_weights.detach().cpu().numpy())
                synch_out_tracking.append(synchronisation_out.detach().cpu().numpy())
                synch_action_tracking.append(synchronisation_action.detach().cpu().numpy())

        if gate_losses:
            self.latest_gate_loss = torch.stack(gate_losses).mean()
        elif gate_supervision_active:
            self.latest_gate_loss = torch.tensor(0.0, device=device)
        else:
            self.latest_gate_loss = None

        if gate_value_count:
            mean_gate_value = gate_value_sum / gate_value_count
            self.latest_gate_metrics = {
                'mean_gate_value': mean_gate_value,
                'num_probes': gate_probe_count,
                'probe_accuracy': (gate_probe_accuracy_sum / gate_probe_count) if gate_probe_count else None,
                'probe_open_frac': (gate_probe_open_sum / gate_probe_count) if gate_probe_count else None,
            }
        else:
            self.latest_gate_metrics = {}

        if gate_history:
            self.latest_gate_sequence = torch.stack(gate_history, dim=-1)
        else:
            self.latest_gate_sequence = None

        # --- Return Values ---
        if track:
            return predictions, certainties, (np.array(synch_out_tracking), np.array(synch_action_tracking)), np.array(pre_activations_tracking), np.array(post_activations_tracking), np.array(attention_tracking)
        return predictions, certainties, synchronisation_out
