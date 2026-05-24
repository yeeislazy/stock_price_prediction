import torch

def test_model(model, test_dataloader, device, parameters, targets_scaler=None):
    val_loss = 0
    test_targets_loss = {target+"_test_loss": 0 for target in parameters["target_columns"]}
    model.eval()
    test_targets_unscaled = {target+"_test_unscaled_mae": 0 for target in parameters["target_columns"]}
    test_targets_direction_acc = {target+"_test_direction_acc": 0 for target in parameters["target_columns"]}

    target_to_idx = {
        target: idx
        for idx, target in enumerate(parameters["target_columns"])
    }
    
    # remaining weight allocation
    weight_decay = parameters.get("weight_decay", 0.7)
    num_targets = len(parameters["target_columns"])
    weights = []
    remaining_weight = 1.0
    for i in range(num_targets-1):
        current_weight = remaining_weight * weight_decay
        weights.append(current_weight)
        remaining_weight -= current_weight
    weights.append(remaining_weight) #eg: for 3 targets with weight_decay=0.7, weights would be [0.7, 0.21, 0.09]

    weights = torch.tensor(weights, device=device)
    
    for X_batch, Y_batch in test_dataloader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        with torch.no_grad():
            outputs = model(X_batch)
            loss_per_target = (outputs - Y_batch) ** 2
            weighted_loss = loss_per_target * weights
            loss = weighted_loss.sum(dim=1).mean()
            val_loss += loss.item()
            
            for target in parameters["target_columns"]:
                target_idx = target_to_idx[target]
                test_targets_loss[target+"_test_loss"] += ((outputs[:, target_idx] - Y_batch[:, target_idx]) ** 2).mean().item()
                
            
            # unscale the outputs and targets to evaluate actual difference and direction accuracy
            if targets_scaler is not None:
                outputs_unscaled = targets_scaler.inverse_transform(outputs.cpu().numpy())
                Y_batch_unscaled = targets_scaler.inverse_transform(Y_batch.cpu().numpy())
                
                for target in parameters["target_columns"]:
                    target_idx = target_to_idx[target]
                    test_targets_unscaled[target+"_test_unscaled_mae"] += (abs(outputs_unscaled[:, target_idx] - Y_batch_unscaled[:, target_idx])).mean().item()
                    
                    test_targets_direction_acc[target+"_test_direction_acc"] += (
                        ((outputs_unscaled[:, target_idx] > 0) == (Y_batch_unscaled[:, target_idx] > 0))
                        .mean()
                    )
                    
                
    avg_val_loss = val_loss / len(test_dataloader)
    avg_test_targets_loss = {target+"_test_loss": test_targets_loss[target+"_test_loss"] / len(test_dataloader) for target in parameters["target_columns"]}
    avg_test_targets_unscaled = {target+"_test_unscaled_mae": test_targets_unscaled[target+"_test_unscaled_mae"] / len(test_dataloader) for target in parameters["target_columns"]}
    avg_test_targets_direction_acc = {target+"_test_direction_acc": test_targets_direction_acc[target+"_test_direction_acc"] / len(test_dataloader) for target in parameters["target_columns"]}
    
    metrics = {"val_loss": avg_val_loss, **avg_test_targets_loss, **avg_test_targets_unscaled, **avg_test_targets_direction_acc}
    
    return metrics