import torch
import math
import torch.nn as nn
import torch.nn.functional as F


def compute_ctc_loss(predictions, targets, blank_label=0):
    """
    Computes the Connectionist Temporal Classification (CTC) loss.

    Args:
        predictions: A tensor of shape [B, C, L] representing the logits of the 
                     predicted sequences.  B is the batch size, C is the number
                     of classes (including the blank label), and L is the sequence
                     length of the predictions.
        targets: A tensor of shape [B, T] representing the target sequences.
                 B is the batch size and T is the target sequence length.
                 Note that T can vary within the batch.
        blank_label: The index of the blank label.  Defaults to 0.

    Returns:
        The CTC loss (a scalar tensor).
    """

    batch_size, num_classes, prediction_length = predictions.shape
    _, target_length = targets.shape

    # 1. Log softmax on predictions:  Crucially, CTC loss requires log probabilities.
    log_probs = F.log_softmax(predictions, dim=1)  # Shape: [B, C, L]

    # 2.  Prepare inputs for torch.nn.CTCLoss:
    #    a.  Convert log_probs to shape (L, B, C):  CTCLoss expects time first.
    log_probs = log_probs.permute(2, 0, 1)  # Shape: [L, B, C]

    #    b.  Get lengths of the predicted sequences (all L in this case).
    input_lengths = torch.full(size=(batch_size,), fill_value=prediction_length, dtype=torch.long)

    #    c.  Get lengths of the target sequences.
    target_lengths = torch.tensor([t.shape[0] for t in targets], dtype=torch.long) # Handle variable target lengths

    # 3. Create the CTCLoss criterion.  `blank=blank_label` is essential!
    ctc_loss = torch.nn.CTCLoss(blank=blank_label, reduction='mean') # 'mean' for averaging over the batch

    # 4. Calculate the loss.  `targets` needs to be a concatenated tensor.
    #    We handle padding by only passing the valid lengths to CTCLoss.
    concatenated_targets = torch.cat(list(targets)) # Concatenate targets

    loss = ctc_loss(log_probs, concatenated_targets, input_lengths, target_lengths)

    return loss

def sort_loss(predictions, targets):
    """
    The sort task was used partly to show that ctc loss can work.
    """
    loss = compute_ctc_loss(predictions, targets, blank_label=predictions.shape[1]-1)
    return loss

def image_classification_loss(predictions,
                              certainties,
                              targets,
                              use_most_certain=True,
                              retentions=None,
                              dwell_start_frac=0.33,
                              lambda_mono=0.01):
    """
    Computes the maze loss with auto-extending cirriculum.

    Predictions are of shape: (B, class, internal_ticks),
    Certainties are of shape: (B, 2, internal_ticks), 
        where the inside dimension (2) is [normalised_entropy, 1-normalised_entropy]
    Targets are of shape: [B]

    use_most_certain will select either the most certain point or the final point. 
    If retentions is provided, a third dwell tick (highest retention after warmup) is included
    and a monotonicity regulariser over r_t is added.
    """
    targets_expanded = torch.repeat_interleave(targets.unsqueeze(-1), predictions.size(-1), -1)
    # Losses are of shape [B, internal_ticks]
    losses = nn.CrossEntropyLoss(reduction='none')(predictions, targets_expanded)
    T = predictions.size(-1)
    batch_size = predictions.size(0)

    loss_index_1 = losses.argmin(dim=1)
    certainty_indices = certainties[:, 1].argmax(-1)
    if use_most_certain:
        where_most_certain = certainty_indices
        loss_index_2 = certainty_indices
    else:
        where_most_certain = torch.full_like(certainty_indices, -1)
        loss_index_2 = torch.full_like(certainty_indices, T - 1)

    tick_indices = [loss_index_1, loss_index_2]

    dwell_index = None
    if retentions is not None:
        dwell_start_frac = float(dwell_start_frac)
        dwell_start = int(math.floor(dwell_start_frac * T))
        dwell_start = max(0, min(T - 1, dwell_start))
        dwell_region = retentions[:, dwell_start:]
        dwell_region = dwell_region if dwell_region.numel() else retentions[:, -1:]
        dwell_relative = dwell_region.argmax(dim=1)
        dwell_index = dwell_relative + dwell_start
        tick_indices.append(dwell_index)

    tick_stack = torch.stack(tick_indices, dim=1)
    selection_mask = losses.new_zeros(batch_size, T)
    batch_indexer = torch.arange(batch_size, device=predictions.device).unsqueeze(1).expand_as(tick_stack)
    selection_mask[batch_indexer, tick_stack] = 1.0
    selection_counts = selection_mask.sum(dim=1).clamp_min(1.0)
    task_loss = ((losses * selection_mask).sum(dim=1) / selection_counts).mean()

    mono_loss = torch.tensor(0.0, device=predictions.device, dtype=losses.dtype)
    if retentions is not None and lambda_mono:
        diffs = torch.relu(retentions[:, :-1] - retentions[:, 1:])
        mono_loss = diffs.sum(dim=1).mean()
        task_loss = task_loss + lambda_mono * mono_loss

    return task_loss, where_most_certain

def maze_loss(predictions, certainties, targets, cirriculum_lookahead=5, use_most_certain=True):
    """
    Computes the maze loss with auto-extending cirriculum.

    Predictions are of shape: (B, route_length, class, internal_ticks),
        where classes are in [0,1,2,3,4] for [Up, Down, Left, Right, Wait]
    Certainties are of shape: (B, 2, internal_ticks), 
        where the inside dimension (2) is [normalised_entropy, 1-normalised_entropy]
    Targets are of shape: [B, route_length]

    cirriculum_lookahead: how far to look ahead in the auto-cirriculum

    use_most_certain will select either the most certain point or the final point. For baselines,
        the final point proved the only usable option. 
    
    """
    # Predictions reshaped to: [B*route_length, 5, internal_ticks]
    predictions_reshaped = predictions.flatten(0,1)
    # Targets reshaped to: [B*route_length, internal_ticks]
    targets_reshaped = torch.repeat_interleave(targets.unsqueeze(-1), 
                                               predictions.size(-1), -1).flatten(0,1).long()
    
    # Losses are of shape [B, route_length, internal_ticks]
    losses = nn.CrossEntropyLoss(reduction='none')(predictions_reshaped, targets_reshaped)
    losses = losses.reshape(predictions[:,:,0].shape)
    
    # Below is the code for auto-cirriculum
    # Find where correct, and make sure to always push +5 beyond that
    iscorrects = (predictions.argmax(2) == targets.unsqueeze(-1)).cumsum(1)
    correct_mask = (iscorrects == torch.arange(1, iscorrects.size(1)+1, device=iscorrects.device).reshape(1, -1, 1))
    correct_mask[:,0,:] = 1
    upto_where = correct_mask.cumsum(1).argmax(1).max(-1)[0]+cirriculum_lookahead
    loss_mask = torch.zeros_like(losses)
    for bi in range(predictions.size(0)):
        loss_mask[bi, :upto_where[bi]] = 1

    # Reduce losses along route dimension
    # Will now be of shape [B, internal_ticks]
    losses = (losses * loss_mask).sum(1)/(loss_mask.sum(1))

    loss_index_1 = losses.argmin(dim=1)
    loss_index_2 = certainties[:,1].argmax(-1)
    if not use_most_certain:
        loss_index_2[:] = -1
    
    batch_indexer = torch.arange(predictions.size(0), device=predictions.device)
    loss_minimum_ce = losses[batch_indexer, loss_index_1]
    loss_selected = losses[batch_indexer, loss_index_2]

    loss = ((loss_minimum_ce + loss_selected)/2).mean()
    return loss, loss_index_2, upto_where.detach().cpu().numpy()

def parity_loss(predictions, certainties, targets, use_most_certain=True):
    """
    Computes the parity loss.

    Predictions are of shape: (B, parity_sequence_length, class, internal_ticks),
        where classes are in [0,1,2,3,4] for [Up, Down, Left, Right, Wait]
    Certainties are of shape: (B, 2, internal_ticks), 
        where the inside dimension (2) is [normalised_entropy, 1-normalised_entropy]
    Targets are of shape: [B, parity_sequence_length]

    use_most_certain will select either the most certain point or the final point. For baselines,
        the final point proved the only usable option. 
    """

    # Losses are of shape [B, parity_sequence_length, internal_ticks]
    losses = nn.CrossEntropyLoss(reduction='none')(predictions.flatten(0,1), 
                                                   torch.repeat_interleave(targets.unsqueeze(-1), 
                                                                           predictions.size(-1), -1).flatten(0,1).long()).reshape(predictions[:,:,0].shape)

    # Average the loss over the parity sequenece dimension
    losses = losses.mean(1)

    loss_index_1 = losses.argmin(dim=1)
    loss_index_2 = certainties[:,1].argmax(-1)
    if not use_most_certain:
        loss_index_2[:] = -1
    
    batch_indexer = torch.arange(predictions.size(0), device=predictions.device)
    loss_minimum_ce = losses[batch_indexer, loss_index_1].mean()
    loss_selected = losses[batch_indexer, loss_index_2].mean()

    loss = (loss_minimum_ce + loss_selected)/2
    return loss, loss_index_2


def qamnist_loss(predictions, certainties, targets, use_most_certain=True):
    """
    Computes the qamnist loss over the last num_answer_steps steps.

    Predictions are of shape: (B, class, internal_ticks),
    Certainties are of shape: (B, 2, internal_ticks), 
        where the inside dimension (2) is [normalised_entropy, 1-normalised_entropy]
    Targets are of shape: [B]
    num_answer_steps: number of steps to consider for the loss

    use_most_certain will select either the most certain point or the final point. 
    """

    losses = nn.CrossEntropyLoss(reduction='none')(predictions, 
                                                   torch.repeat_interleave(targets.unsqueeze(-1), predictions.size(-1), -1))
        
    loss_index_1 = losses.argmin(dim=1)
    loss_index_2 = certainties[:,1].argmax(-1)
    if not use_most_certain:
        loss_index_2[:] = -1
    
    batch_indexer = torch.arange(predictions.size(0), device=predictions.device)
    loss_minimum_ce = losses[batch_indexer, loss_index_1].mean()
    loss_selected = losses[batch_indexer, loss_index_2].mean()

    loss = (loss_minimum_ce + loss_selected)/2
    return loss, loss_index_2
