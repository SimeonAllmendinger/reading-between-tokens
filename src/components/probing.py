import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from torch import nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from schemas import ValueProbe, Dataset, Metrics

from settings import SETTINGS, LOGGER

label_encoder = LabelEncoder()


class ValueProbeModel(nn.Module):
    """
    A simple feedforward neural network for value probing.
    """
    
    def __init__(self, input_dim: int, 
                 output_dim: int,
                 n_epochs: int):
        
        super(ValueProbeModel, self).__init__()
        
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.n_epochs = n_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
        self.linear.apply(self.init_weights)
    
    @staticmethod
    def init_weights(model):
        if isinstance(model, nn.Linear):
            nn.init.kaiming_uniform_(model.weight, nonlinearity='linear')
            nn.init.zeros_(model.bias)
        
    def train_model(self, x, y, pos_weight=1.0, early_stopping=False, x_test=None, y_test=None):
        self.train()
        x, y = x.to(self.device), y.to(self.device)
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs)
        
        for epoch in tqdm(range(self.n_epochs), desc=f"Training Value Probe"):
            optimizer.zero_grad()
            
            logits = self.forward(x)
            loss = criterion(logits.squeeze(1), y)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if early_stopping and loss.item() < 0.001:
                LOGGER.warning(f"Early Stopping at Epoch: {epoch}")
                break

            # log the loss in tqdm description
            if epoch % 1000 == 0:
                tqdm.write(f"Epoch: {epoch}, Loss: {loss.item()}")
                if x_test is not None and y_test is not None:
                    _ = self.evaluate(x_test, y_test)
    
    def forward(self, x):
        logits = self.dropout(self.linear(x))
        return logits
    
    @torch.no_grad()
    def predict(self, x):
        self.eval()
        x = x.to(self.device)
        logits = self.linear(x.to(self.device))
        probs = torch.sigmoid(logits).cpu()
        
        # get all indices where the value is greater than 0.5
        y = torch.where(
            probs.squeeze() > 0.5,
            torch.tensor(1),
            torch.tensor(0)
        )
        return y
    
    @torch.no_grad()
    def evaluate(self, x, y):
        # Move x and y to CPU
        y_pred = self.predict(x).cpu().numpy()
        y = y.cpu().numpy()
        
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "f1": f1_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
        }
        
        LOGGER.info(f"Metrics: {metrics}")
        
        return metrics
        
    def get_linear_weights_and_bias(self):
        return self.linear.weight.cpu().detach(), self.linear.bias.cpu().detach()


def get_value_probes(
        dataset: Dataset,
        path_value_probe_dir: str
    ) -> list[ValueProbe]:
    """
    Get the probing results.
    
    Args:
        dataset: Dataset
            The dataset.
        path_value_probe_dir: str
            The path to the value probe directory.
    
    Returns:
        probing_results: list[ValueProbe]
            A list of value probes.
    """
    value_probes = []
    
    label_encoder.fit(dataset.issue_owner_names)
    
    for _model_id in dataset.model_ids:
        
        # X and y are initially empty tensors
        X_train_list = []
        y_train_list = []
        X_test_list = []
        y_test_list = []
        
        for _data_item in tqdm(dataset.data, desc="Build probing dataset"):
            
            if _data_item.model_id != _model_id:
                continue
            
            activation = torch.load(_data_item.path_activation_file, weights_only=False)
                
            # Collect accum activation tensor in a list for last layer
            layer_min = int(round(activation.shape[0] * SETTINGS.probing['Value_Probe']['activation_layer']['min_ratio'],0))
            layer_max = int(round(activation.shape[0] * SETTINGS.probing['Value_Probe']['activation_layer']['max_ratio'],0))
            
            x_i = activation[layer_min:layer_max, :].clone().to(torch.float32).unsqueeze(0)

            # Process the label
            mapped_io_name = _data_item.issue_owner.get_name_mapped(name_mapping=dataset.issue_config['name_mapping'])
            y_i = [torch.tensor(label_encoder.transform([mapped_io_name]), dtype=torch.float32)] * x_i.shape[1]
            
            # Collect training data
            X_train_list.extend(x_i)
            y_train_list.extend(y_i)

        # If there are no test items, we split the training data into train and test
        
        X_train, X_test, y_train, y_test = train_test_split(
            torch.cat(X_train_list),
            torch.cat(y_train_list),
            test_size=SETTINGS.probing['Value_Probe']['test_size'],
            random_state=SETTINGS.probing['Value_Probe']['random_state'],
            shuffle=SETTINGS.probing['Value_Probe']['shuffle'],
            )
        
        LOGGER.info(f"Layer Min: {layer_min}, Layer Max: {layer_max}")
        LOGGER.info(f"X train shape: {X_train.shape}, y train shape: {y_train.shape}")
        LOGGER.info(f"X test shape: {X_test.shape}, y test shape: {y_test.shape}")
        
        # train different probes for each issue owner in the dataset
        all_ios = dataset.get_all_unique_issue_owners(model_id=_model_id)
        LOGGER.info(f"Number of Issue Owners: {len(all_ios)}")
        
        for _io in tqdm(all_ios, desc="Training Value Probes"):
            
            # y_train_party is 1 if y_train == label_encoder.transform([party_name]), 0 otherwise
            mapped_io_name = _io.get_name_mapped(name_mapping=dataset.issue_config['name_mapping'])
            LOGGER.info(f"Issue Owner: {mapped_io_name}")
            
            y_train_io = torch.tensor([1 if y_i == torch.tensor(label_encoder.transform([mapped_io_name])) else 0 for y_i in y_train], dtype=torch.float32)
            y_test_io = torch.tensor([1 if y_i == torch.tensor(label_encoder.transform([mapped_io_name])) else 0 for y_i in y_test], dtype=torch.float32)
            
            # Log number of ones in y_train, y_test
            LOGGER.info(f"Number of ones in y_train_party: {torch.sum(y_train_io)} with length {len(y_train_io)}")
            LOGGER.info(f"Number of ones in y_test_party: {torch.sum(y_test_io)} with length {len(y_test_io)}")
            
            probe_model = ValueProbeModel(
                input_dim=(X_train.shape[1]),
                output_dim=1,
                n_epochs=SETTINGS.probing["Value_Probe"]["n_epochs"],
                )
            
            # create a weighting scheme for the training data so that the zeros and 1 in y_train_party are balanced
            pos_weight = (len(y_train_io) - torch.sum(y_train_io)) / torch.sum(y_train_io).clone().detach()
            LOGGER.info(f"Positive Weight: {pos_weight}")
            
            # Train the probe
            probe_model.train_model(
                X_train,
                y_train_io,
                pos_weight=pos_weight,
                x_test=X_test,
                y_test=y_test_io
            )
            
            # Evaluate the probe
            metrics = probe_model.evaluate(X_test, y_test_io)
            
            metrics = Metrics(**metrics)
            
            value_probe = ValueProbe(
                model_id=_model_id,
                issue_owner=_io,
                metrics=metrics,
                layer_min=layer_min,
                layer_max=layer_max,
                path_value_probe_dir=path_value_probe_dir,
                country=dataset.country,
            )
            
            torch.save(probe_model, value_probe.path_value_probe)
            
            value_probes.append(value_probe)
        
        # Calculate the similarity between the value probes
        compute_value_probes_similarity(value_probes, X_train, y_train, layer_min, layer_max)
        
    return value_probes


@torch.no_grad()
def compute_value_probes_similarity(
    value_probes: list[ValueProbe],
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    layer_min: int,
    layer_max: int
    ):
    """
    Compute the similarity between the value probes with cosine similarity.
    
    Args:
        value_probes: list[ValueProbe]
            A list of value probes.
        X_train: torch.Tensor
            The training data.
        y_train: torch.Tensor
            The training labels.
        layer_min: int
            The minimum layer index for the value probe.
        layer_max: int
            The maximum layer index for the value probe.
    
    Returns:
        None, but updates the similarity attribute of each ValueProbe.
        Also visualizes the similarity matrix and a PCA/TSNE plot.
    """
    
    similarity_matrix = torch.zeros((len(value_probes), len(value_probes)))
    
    LOGGER.info(f"Computing Value Probes Similarity Matrix")
    
    for i, value_probe in enumerate(value_probes):
        for j, _value_probe in enumerate(value_probes):
            
            if j <= i:
                continue
            
            probe_model = torch.load(value_probe.path_value_probe, weights_only=False)
            _probe_model = torch.load(_value_probe.path_value_probe, weights_only=False)
            
            probe_model_weights, _ = probe_model.get_linear_weights_and_bias()
            _probe_model_weights, _ = _probe_model.get_linear_weights_and_bias()
            
            cos_sim = torch.nn.functional.cosine_similarity(
                probe_model_weights.squeeze(),
                _probe_model_weights.squeeze(),
                dim=0
            )
            
            value_probe_name = value_probe.issue_owner.name + "_" + value_probe.model_id
            _value_probe_name = _value_probe.issue_owner.name + "_" + _value_probe.model_id
            
            value_probe.similarity[_value_probe_name] = cos_sim.item()
            _value_probe.similarity[value_probe_name] = cos_sim.item()
            
            similarity_matrix[i, j] = cos_sim.item()
            similarity_matrix[j, i] = cos_sim.item()
    
    LOGGER.info("Visualizing Value Probes Similarity Matrix")
    
    # Visualize the similarity matrix and a PCA/TSNE plot as subfigures
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot the similarity matrix
    ax.imshow(similarity_matrix, cmap='viridis')
    ax.set_title("Value Probes Similarity Matrix")
    ax.set_xlabel("Value Probes")
    ax.set_ylabel("Value Probes")
    ax.set_xticks(range(len(value_probes)))
    ax.set_yticks(range(len(value_probes)))
    ax.set_xticklabels([value_probe.issue_owner.name for value_probe in value_probes], rotation=45)
    ax.set_yticklabels([value_probe.issue_owner.name for value_probe in value_probes])
    
    path_savefig= os.path.join(
        "results",
        value_probe.country.lower(),
        value_probe.model_id.replace('/','-'),
        "value_probes",
        f"cosine_similarity.png"
    )
    
    # Check if directory exists:
    if not os.path.exists(os.path.dirname(path_savefig)):
        os.makedirs(os.path.dirname(path_savefig))
    
    # Save the figure
    fig.savefig(path_savefig)
    plt.close(fig)
    
    all_probes = torch.cat(
        [torch.load(value_probe.path_value_probe, weights_only=False).get_linear_weights_and_bias()[0] for value_probe in value_probes]
    )

    pca = PCA(n_components=all_probes.shape[0])
    tsne = TSNE(n_components=2, random_state=42)
    
    # Number of layers to consider for PCA/TSNE
    n_layers = layer_max - layer_min
    n_activations_per_layer = X_train.shape[0] // n_layers
    
    # Plot the PCA plot
    fig, ax = plt.subplots(2, -(-n_layers // 2), figsize=(120, 60))
    country = value_probes[0].country.upper()
    color_mapping = SETTINGS.dataset[country]['color_mapping']
    
    for i in tqdm(range(n_layers), desc="Plotting PCA/TSNE for each layer"):
        # Get the activations for the current layer
        layer_activations = X_train[i * n_activations_per_layer:(i + 1) * n_activations_per_layer, :]
        layer_labels = y_train[i * n_activations_per_layer:(i + 1) * n_activations_per_layer]
        
        # Fit PCA on the layer activations
        layer_probes_pca = pca.fit_transform(all_probes.cpu().numpy())
        layer_activations_pca = pca.transform(layer_activations.cpu().numpy())
    
        layer_activations_probes_pca = np.concatenate((layer_activations_pca, layer_probes_pca), axis=0)

        # Fit TSNE on the PCA transformed activations
        layer_activations_tsne = tsne.fit_transform(layer_activations_probes_pca)
        
        layer_labels_int = layer_labels.cpu().numpy().astype(int)
        layer_labels_names = label_encoder.inverse_transform(layer_labels_int)
        
        # Create color array for each label
        colors = [color_mapping[label] for label in layer_labels_names]
        
        # Plot the PCA results
        ax[i % 2, i // 2].scatter(
            layer_activations_tsne[:len(layer_activations), 0],
            layer_activations_tsne[:len(layer_activations), 1],
            c=colors,
            marker=".",
            s=50,  # Make markers smaller
            alpha=0.8,
            label='Activations'
        )
        
        # Overlay value probes
        for j, value_probe in enumerate(value_probes):
            ax[i % 2, i // 2].scatter(
                layer_activations_tsne[len(layer_activations) + j, 0],
                layer_activations_tsne[len(layer_activations) + j, 1],
                marker='x',
                s=100,  # Make markers larger
                c=color_mapping[value_probe.issue_owner.name_mapped],
                label=value_probe.issue_owner.short_name,
                linewidth=2
            )
            
        ax[i % 2, i // 2].set_title(f"Layer {i + layer_min} PCA/TSNE")
        ax[i % 2, i // 2].set_xlabel("PCA Component 1")
        ax[i % 2, i // 2].set_ylabel("PCA Component 2")
        ax[i % 2, i // 2].legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    
    # Adjust layout
    plt.tight_layout()
    
    # Set the title for the entire figure
    fig.suptitle(f"TSNE Plot of Value Probes for {value_probe.country.upper()} - {value_probe.model_id}", fontsize=16)
    
    # Save the figure
    path_savefig= os.path.join(
        "results",
        value_probe.country.lower(),
        value_probe.model_id.replace('/','-'),
        "value_probes",
        f"tsne.png"
    )
    
    if not os.path.exists(os.path.dirname(path_savefig)):
        os.makedirs(os.path.dirname(path_savefig))
    
    # Save the figure
    fig.savefig(path_savefig)
