import torch
import torch.nn.functional as F

def create_ensemble_predictions(cv_model, ts_model, gnn_model, graph_data, X_seq, satellite_images, company_df):
    predictions = []

    # Get GNN predictions
    gnn_model.eval()
    with torch.no_grad():
        gnn_pred = gnn_model(graph_data.x, graph_data.edge_index)
        gnn_probs = F.softmax(gnn_pred, dim=1)[:, 1]  # Probability of disruption

    # Get Time Series predictions (using last sequence for each company)
    ts_model.eval()
    with torch.no_grad():
        ts_pred = ts_model(torch.FloatTensor(X_seq[-len(company_df):]))
        ts_probs = ts_pred.squeeze()

    # Get CV predictions (using latest images)
    cv_model.eval()
    cv_probs = []
    with torch.no_grad():
        for i in range(len(company_df)):
            # Find image for this company (simplified)
            company_images = [img for img in satellite_images if img['company_id'] == i]
            if company_images:
                img_array = company_images[0]['image_array']
                img_tensor = torch.FloatTensor(img_array).permute(2, 0, 1).unsqueeze(0) / 255.0
                cv_prob = cv_model(img_tensor).item()
                cv_probs.append(cv_prob)
            else:
                cv_probs.append(0.5)  # Default

    cv_probs = torch.FloatTensor(cv_probs)

    # Ensemble predictions (weighted average)
    ensemble_probs = 0.4 * gnn_probs + 0.4 * ts_probs + 0.2 * cv_probs

    return ensemble_probs, gnn_probs, ts_probs, cv_probs
