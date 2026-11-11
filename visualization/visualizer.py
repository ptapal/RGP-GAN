import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.manifold import TSNE
from umap import UMAP 
from pyriemann.utils.distance import distance_riemann
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D 
from scipy.stats import pearsonr


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm",
    "grid.color": "#b0b0b0",     # neutral gray color for the grid
    "grid.linestyle": ":", 
    "grid.alpha": 0.9
})

# apply a clean, consistent style once
try:
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper") # "paper", "notebook", "talk", "poster"
    sns.set_palette("colorblind")
except Exception as e:
    print(f"Could not apply seaborn style: {e}. Using Matplotlib default.")

class BaseVisualizer:
    """Base class for visualization utilities"""
    def __init__(self, output_dir="./visualizations", font_size=17, config=None):
        self.output_dir = output_dir
        self.config = config
        os.makedirs(self.output_dir, exist_ok=True)
        self.dpi = 300
        self.font_size = font_size
        self.font_size_title = font_size + 2

    def save_figure(self, fig, name_prefix, name_suffix=""):
        """Save figure to output directory"""
        save_path = os.path.join(self.output_dir, f"{name_prefix}_{name_suffix}.png") # or pdf
        #fig.tight_layout()
        fig.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        print(f"Figure saved to {save_path}")
        plt.close(fig)
        return save_path

class ModelVisualizer(BaseVisualizer):
    """Visualizer for model-related outputs"""

    def visualize_training_history(self, history, name_prefix="training_history"):
        """Visualize training and validation history."""
        epochs = history.get('epoch', range(len(history.get('loss', []))))
        
        fig, ax1 = plt.subplots(figsize=(8, 5))
        
        # plot Total Loss
        if 'loss' in history:
            ax1.plot(epochs, history['loss'], color='tab:blue', label='Total Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        # plot individual losses if available
        if 'recon_loss' in history:
            ax1.plot(epochs, history['recon_loss'], color='tab:green', linestyle='--', label='Recon Loss', linewidth=1.5)
        if 'kl_loss' in history:
            ax1.plot(epochs, history['kl_loss'], color='tab:red', linestyle=':', label='KL Loss', linewidth=1.5)
        if 'diversity_loss' in history and history.get('diversity_loss'): # check if not empty
             ax1.plot(epochs, history['diversity_loss'], color='tab:purple', linestyle='-.', label='Diversity Loss', linewidth=1.5)

        ax1.legend(loc='upper right')
        #ax1.set_title(f'{name_prefix.replace("_", " ").title()} - Losses')
        ax1.grid(True, linestyle=':', alpha=0.7)
        
        if 'lr' in history and history['lr']:
            ax2 = ax1.twinx()
            ax2.plot(epochs, history['lr'], color='tab:orange', linestyle='--', label='Learning Rate', linewidth=1.5)
            ax2.set_ylabel('Learning Rate', color='tab:orange')
            ax2.tick_params(axis='y', labelcolor='tab:orange')
            ax2.set_yscale('log')
            fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=4)

        print("VISUALIZE TRAINING HISTORY!!!!!!!!!!!!!!!!!!!!!!!!")
        self.save_figure(fig, name_prefix, "_losses_and_lr")
        return os.path.join(self.output_dir, f"{name_prefix}_losses_and_lr.pdf")


    def visualize_latent_space(self, model, X_data, y_labels, device, name_prefix="latent_space", dim_reduction='umap', n_dim=2):
        """Visualize 2D or 3D latent space using UMAP (default) or t-SNE"""
        if not isinstance(X_data, torch.Tensor): X_data = torch.tensor(X_data, dtype=torch.float32)
        if not isinstance(y_labels, torch.Tensor): y_labels = torch.tensor(y_labels, dtype=torch.long)
        
        X_data, y_labels = X_data.to(device), y_labels.to(device)
        model.eval()
        
        latent_vecs_list = []
        with torch.no_grad():
            batch_size = 64
            for i in range(0, len(X_data), batch_size):
                batch_X = X_data[i:i+batch_size]
                encode_output = model.encode(batch_X)
                mu = encode_output[0] if isinstance(encode_output, tuple) else encode_output
                latent_vecs_list.append(mu.cpu().numpy())
        
        latent_vecs = np.vstack(latent_vecs_list)
        labels_np = y_labels.cpu().numpy()
        unique_labels = np.unique(labels_np)

        if len(latent_vecs) <= n_dim or len(latent_vecs) < 5 : # reducer fail with too few samples
            print(f"Not enough samples ({len(latent_vecs)}) for {dim_reduction} {n_dim}D visualization. Skipping.")
            return None

        if dim_reduction == 'umap':
            reducer = UMAP(n_components=n_dim, n_neighbors=min(15, len(latent_vecs)-1), min_dist=0.1, random_state=42)
        elif dim_reduction == 'tsne':
            reducer = TSNE(n_components=n_dim, perplexity=min(30, len(latent_vecs)-1), init='pca', learning_rate='auto', random_state=42)
        else:
            raise ValueError("Unsupported dimensionality reduction method. Choose 'umap' or 'tsne'.")
            
        reduced_vecs = reducer.fit_transform(latent_vecs)
        
        fig = plt.figure(figsize=(8, 6) if n_dim==2 else (9,7))
        if n_dim == 2:
            ax = fig.add_subplot(111)
            sns.scatterplot(x=reduced_vecs[:, 0], y=reduced_vecs[:, 1], hue=labels_np, 
                            palette="viridis", legend='full', ax=ax, s=50, alpha=0.8)
            ax.set_xlabel(f'{dim_reduction.upper()} Component 1')
            ax.set_ylabel(f'{dim_reduction.upper()} Component 2')
        elif n_dim == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(reduced_vecs[:, 0], reduced_vecs[:, 1], reduced_vecs[:, 2], 
                       c=labels_np, cmap="viridis", s=50, alpha=0.8)
            ax.set_xlabel(f'{dim_reduction.upper()} Comp 1')
            ax.set_ylabel(f'{dim_reduction.upper()} Comp 2')
            ax.set_zlabel(f'{dim_reduction.upper()} Comp 3')
        
        ax.set_title(f'{n_dim}D {dim_reduction.upper()} Latent Space')
        if len(unique_labels) > 1: # add legend only if multiple classes
            handles, legend_labels = ax.get_legend_handles_labels() if n_dim==2 else (None, None) # Get handles for 2D
            if handles: # sns scatterplot creates legend items
                ax.legend(title="Class")
            elif n_dim==3 : # legend for 3D scatter
                 legend_elements = [plt.scatter([],[], color=plt.cm.viridis(i/max(1,len(unique_labels)-1)), label=f'Class {ul}') for i, ul in enumerate(unique_labels)]
                 ax.legend(handles=[le[0] if isinstance(le, list) else le for le in legend_elements], title="Class")

        return self.save_figure(fig, name_prefix, f"_{dim_reduction}_{n_dim}d")

    def visualize_reconstructions(self, model, X_data, device, n_samples=3, name_prefix="reconstructions"):
        if not isinstance(X_data, torch.Tensor): X_data = torch.tensor(X_data, dtype=torch.float32)
        X_data = X_data.to(device)
        
        if len(X_data) == 0:
            print("No data provided for reconstructions. Skipping.")
            return None
        
        indices = np.random.choice(len(X_data), min(n_samples, len(X_data)), replace=False)
        samples_X = X_data[indices]
        
        model.eval()
        with torch.no_grad():
            outputs = model(samples_X) # RVAE forward: recon_x, z, mu, logvar, h_enc, h_dec
            recon_batch = outputs[0] if isinstance(outputs, tuple) else outputs

        fig, axes = plt.subplots(len(samples_X), 3, figsize=(12, 3 * len(samples_X)), squeeze=False)
        
        for i in range(len(samples_X)):
            orig_np = samples_X[i].cpu().numpy()
            recon_np = recon_batch[i].cpu().numpy()
            diff_np = orig_np - recon_np
            
            common_min = min(orig_np.min(), recon_np.min())
            common_max = max(orig_np.max(), recon_np.max())

            im = axes[i, 0].imshow(orig_np, cmap='viridis', vmin=common_min, vmax=common_max)
            axes[i, 0].set_title(f'Original {i+1}')
            fig.colorbar(im, ax=axes[i, 0], orientation='vertical', fraction=0.046, pad=0.04)

            im = axes[i, 1].imshow(recon_np, cmap='viridis', vmin=common_min, vmax=common_max)
            axes[i, 1].set_title(f'Reconstruction {i+1}')
            fig.colorbar(im, ax=axes[i, 1], orientation='vertical', fraction=0.046, pad=0.04)
            
            diff_limit = np.abs(diff_np).max()
            im = axes[i, 2].imshow(diff_np, cmap='coolwarm', vmin=-diff_limit, vmax=diff_limit)


            try:
                dist = distance_riemann(orig_np, recon_np)
                axes[i, 2].set_title(f'Difference (Dist: {dist:.2f})')
            except:
                axes[i, 2].set_title('Difference')
            fig.colorbar(im, ax=axes[i, 2], orientation='vertical', fraction=0.046, pad=0.04)

            for ax_col in range(3): axes[i, ax_col].axis('off')

        return self.save_figure(fig, name_prefix)

    def visualize_latent_space_with_subjects(self, model, X_data, y_class_labels, subject_ids_array, device, 
                                           name_prefix="latent_space_subjects", filter_class_display=None, dim_reduction='umap'):
        """Visualize 2D latent space colored by subject, optionally filtered by class"""

        plt.rcParams.update({'font.size': self.font_size})

        if not isinstance(X_data, torch.Tensor): X_data = torch.tensor(X_data, dtype=torch.float32)
        if not isinstance(y_class_labels, torch.Tensor): y_class_labels = torch.tensor(y_class_labels, dtype=torch.long)
        
        # filter data first if a specific class display is requested
        if filter_class_display is not None:
            mask = (y_class_labels == filter_class_display).cpu().numpy()
            X_data = X_data[mask]
            y_class_labels = y_class_labels[mask] # will all be filter_class_display
            subject_ids_array = subject_ids_array[mask]
            if len(X_data) == 0:
                print(f"No data for class {filter_class_display} to visualize with subjects.")
                return None
        
        X_data = X_data.to(device)
        model.eval()
        
        latent_vecs_list = []
        with torch.no_grad():
            batch_size = 64
            for i in range(0, len(X_data), batch_size):
                batch_X = X_data[i:i+batch_size]
                encode_output = model.encode(batch_X)
                mu = encode_output[0] if isinstance(encode_output, tuple) else encode_output
                latent_vecs_list.append(mu.cpu().numpy())
        
        latent_vecs = np.vstack(latent_vecs_list)
        
        if len(latent_vecs) < 5 :
            print(f"Not enough samples ({len(latent_vecs)}) for subject latent space visualization.")
            return None

        if dim_reduction == 'umap':
            reducer = UMAP(n_components=2, n_neighbors=min(15, len(latent_vecs)-1), min_dist=0.1, random_state=42)
        elif dim_reduction == 'tsne':
            reducer = TSNE(n_components=2, perplexity=min(30, len(latent_vecs)-1), init='pca', learning_rate='auto', random_state=42)
        else:
            raise ValueError("Unsupported method. Choose 'umap' or 'tsne'.")

        reduced_vecs = reducer.fit_transform(latent_vecs)
        
        fig, ax = plt.subplots(figsize=(10, 7))
        unique_subjects = np.unique(subject_ids_array)
        subject_palette = sns.color_palette("husl", n_colors=len(unique_subjects))
        
        scatter = sns.scatterplot(x=reduced_vecs[:, 0], y=reduced_vecs[:, 1], hue=subject_ids_array, 
                                  palette=subject_palette, legend='full', ax=ax, s=60, alpha=0.9)
        
        #title = f'2D {dim_reduction.upper()} Latent Space by Subject'
        #if filter_class_display is not None:
        #    title+= f'(Class {filter_class_display} Data Only)'
        #ax.set_title(title)
        ax.set_xlabel(f'{dim_reduction.upper()} Component 1')
        ax.set_ylabel(f'{dim_reduction.upper()} Component 2')
        ax.legend(title="Subject ID", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        suffix = f"_class_filt_{filter_class_display}" if filter_class_display is not None else "_all_classes"
        return self.save_figure(fig, name_prefix, f"{suffix}_{dim_reduction}_2d")

    def visualize_cross_class_separability(self, model_to_test_with, X_in_dist, X_out_dist,
                                        class_labels, device, name_prefix="separability_test"):
        """
        Performs a diagnostic test by encoding data with a single class-specific model to check for separability.

        Args:
            model_to_test_with (torch.nn.Module): A model trained on a single class
            X_in_dist (np.ndarray): Data from the class the model was trained on
            X_out_dist (np.ndarray): Data from a different, unseen class
            class_labels (tuple): e.g., (0, 1), the labels for the two classes
            device (torch.device): The device for tensor operations
            name_prefix (str): A prefix for the saved figure file
        """
        model_to_test_with.eval()

        # get latent vectors
        def encode_data(X_data):
            with torch.no_grad():
                return model_to_test_with.encode(torch.tensor(X_data, dtype=torch.float32).to(device))[0].cpu().numpy()

        # encode both datasets using the SAME model
        latent_in_dist = encode_data(X_in_dist)
        latent_out_dist = encode_data(X_out_dist)

        # combine the latent vectors and create corresponding labels
        combined_latent_vecs = np.vstack([latent_in_dist, latent_out_dist])
        labels = np.array([class_labels[0]] * len(latent_in_dist) + [class_labels[1]] * len(latent_out_dist))

        # reduce dimensionality with UMAP
        reducer = UMAP(n_components=2, n_neighbors=min(15, len(combined_latent_vecs)-1), random_state=42)
        reduced_vecs = reducer.fit_transform(combined_latent_vecs)

        # plottt
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.scatterplot(x=reduced_vecs[:, 0], y=reduced_vecs[:, 1], hue=labels,
                        palette='viridis', legend='full', ax=ax, s=50, alpha=0.8)

        ax.set_title(f'Latent Space of VAE Trained on Class {class_labels[0]}', fontsize=self.font_size_title)
        ax.set_xlabel('UMAP Component 1', fontsize=self.font_size)
        ax.set_ylabel('UMAP Component 2', fontsize=self.font_size)
        ax.legend(title="Original Class Label")
        ax.grid(True, linestyle=':', alpha=0.6)

        self.save_figure(fig, name_prefix)

class EvaluationVisualizer(BaseVisualizer):
    """Visualizer for evaluation results"""

    def visualize_improvement_distribution(self, results_data: dict, experiment_name: str):
        """
        For each generator, creates a detailed figure with two subplots:
        1. A box plot for the accuracy improvement of (Baseline - Augmented).
        2. A box plot for the accuracy improvement of (Baseline - Synthetic).

        This plot is only generated for LOSO evaluations!

        Args:
            results_data (dict): The full results dictionary from the evaluator
            experiment_name (str): The base name for saving the figures
        """
        # data extraction and validationn
        baseline_results = results_data.get('baseline_results', {})
        classifiers = list(baseline_results.get('average_metrics', {}).keys())

        if 'fold_scores' not in baseline_results:
            print("No improvement distribution plot: 'fold_scores' not found in results (probably not a LOSO evaluation).")
            return
        if not classifiers:
            print("Warning: No classifier data found in results. No improvement distribution visualization.")
            return

        subject_perf = results_data.get('subject_performance')
        
        all_subjects = sorted([int(s) for s in subject_perf.get('baseline', {}).keys()])
        palette = sns.color_palette("husl", n_colors=len(all_subjects))
        subject_color_map = {subject: palette[i] for i, subject in enumerate(all_subjects)}

        def _plot_subplot(ax, improvements_per_clf, title):
            mean_improvements = [np.mean(imp) for imp in improvements_per_clf]
            box_width = 0.6

            ax.boxplot(improvements_per_clf,
                            widths=box_width,
                            showfliers=False,
                            patch_artist=True,
                            showmeans=False,
                            medianprops=dict(color="blue", linewidth=1.5),
                            boxprops=dict(edgecolor='black', linewidth=1.5, facecolor='none'),
                            whiskerprops=dict(linestyle='--', color='black', linewidth=1.5),
                            capprops=dict(color='black', linewidth=1.5)
                        )


            for i, improvements in enumerate(improvements_per_clf): # i = classifier index
                # loop over each subject's score for the current classifier
                for j, score in enumerate(improvements): # j = subject index
                    # get the subject ID and their specific color
                    subject_id = all_subjects[j]
                    color = subject_color_map.get(subject_id)

                    jitter_x = np.random.normal(i + 1, 0.04)
                    # plot the single, colored dot
                    ax.scatter(jitter_x, score,
                            facecolors=color,        
                            edgecolors='none',      
                            s=65,                    
                            alpha=0.7,               
                            zorder=10)

            for i, mean_val in enumerate(mean_improvements):
                line_width = box_width
                ax.plot([i + 1 - line_width / 2, i + 1 + line_width / 2], [mean_val, mean_val],
                        color='red', linestyle='-', linewidth=1.5)
                ax.text(i + 1 + line_width / 2 + 0.02, mean_val, f'{mean_val * 100:+.1f}%', color='red',
                        va='center', ha='left', fontweight='bold', fontsize=self.font_size - 8)

            ax.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
            ax.set_title(title, fontsize=self.font_size)
            ax.set_xticks(np.arange(len(classifiers)) + 1)
            ax.set_xticklabels([clf.upper() for clf in classifiers], fontsize=self.font_size)
            ax.tick_params(axis='y', labelsize=self.font_size - 5)

        for gen_type in self.config.generator.generator_types:
            gen_results = results_data['per_generator_results'].get(gen_type, {})
            if not gen_results:
                continue

            print(f"--- Visualizing Improvement Distribution for Generator: {gen_type.upper()} ---")

            # cool -> using sharey=True links the axes so the scales are identical
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
            #fig.suptitle(f'Improvement Distribution By Classifier For {gen_type.capitalize()} Generator', fontsize=self.font_size_title, y=0.95)

            # prepare data for both subplots
            improvements_aug, improvements_synth = [], []
            for clf in classifiers:
                base_scores = np.array(baseline_results.get('fold_scores', {}).get(clf, []))
                aug_scores = np.array(gen_results.get('fold_scores', {}).get('aug',{}).get(clf, []))
                synth_scores = np.array(gen_results.get('fold_scores', {}).get('synthetic', {}).get(clf, []))

                if base_scores.size > 0:
                    if aug_scores.size > 0:
                        improvements_aug.append(aug_scores - base_scores)
                    if synth_scores.size > 0:
                        improvements_synth.append(synth_scores - base_scores)

            percent_formatter = FuncFormatter(lambda y, pos: f'{y * 100:.0f}')
            axes[0].yaxis.set_major_formatter(percent_formatter)
            
            # update the Y-axis label to reflect the new units
            axes[0].set_ylabel('Accuracy Improvement (%)', fontsize=self.font_size)
            axes[1].tick_params(axis='y', labelleft=True)

            # plot each subplot using the helper
            _plot_subplot(axes[0], improvements_aug, 'Improvement (Baseline - Augmented)')
            _plot_subplot(axes[1], improvements_synth, 'Improvement (Baseline - Synthetic)')

            if subject_color_map:
                legend_elements = [Line2D([0], [0], marker='o', color='w',
                                        label=f'Subject {s}',
                                        markerfacecolor=c, markersize=8,
                                        markeredgecolor='none')
                                for s, c in subject_color_map.items()]

                # place legend horizontally at the bottom of the figure
                fig.legend(handles=legend_elements,
                        loc='lower center',             # positioned at the bottom
                        bbox_to_anchor=(0.5, -0.05),    # below the plot area
                        ncol=min(len(all_subjects), 6), # arrange in columns
                        fontsize=self.font_size - 4,
                        title="Subjects",
                        title_fontsize=self.font_size
                        )
            
            # make room for the legend at the bottom
            fig.tight_layout(rect=[0, 0.1, 1, 0.93]) # increased bottom margin
                    
            self.save_figure(fig, experiment_name, f"improvement_distribution_{gen_type}")
            

    def visualize_performance_distribution(self, results_data: dict, experiment_name: str):
        """
        For each generator, create a figure showing the absolute performance
        distributions for Baseline, Augmented, and Synthetic conditions.

        This plot is only generated for LOSO evaluations.

        Args:
            results_data (dict): The full results dictionary from the evaluator
            experiment_name (str): The base name for saving the figures
        """
        # data extraction
        baseline_results = results_data.get('baseline_results', {})
        classifiers = list(baseline_results.get('average_metrics', {}).keys())
        if not classifiers: return

        subject_perf = results_data.get('subject_performance')
        if not subject_perf or not subject_perf.get('baseline'):
            all_subjects, subject_color_map = [], {}
        else:
            all_subjects = sorted([int(s) for s in subject_perf.get('baseline', {}).keys()])
            palette = sns.color_palette("husl", n_colors=len(all_subjects))
            subject_color_map = {subject: palette[i] for i, subject in enumerate(all_subjects)}

        # --- create one figure per generator ---
        for gen_type in self.config.generator.generator_types:
            gen_results = results_data['per_generator_results'].get(gen_type, {})
            if not gen_results: continue

            # set up one subplot PER CLASSIFIER
            fig, axes = plt.subplots(1, len(classifiers), figsize=(8 * len(classifiers), 7), sharey=True)

            #fig.suptitle(f'Performance Distribution by Condition Using {gen_type.capitalize()} Generator',
                        #fontsize=self.font_size_title, y=1.0)

            # loop through each classifier to create its subplot
            for i, clf in enumerate(classifiers):
                ax = axes[i]

                # GATHER RAW PERFORMANCE SCORES
                base_scores = np.array(baseline_results.get('fold_scores', {}).get(clf, []))
                aug_scores = np.array(gen_results.get('fold_scores', {}).get('aug', {}).get(clf, []))
                synth_scores = np.array(gen_results.get('fold_scores', {}).get('synthetic', {}).get(clf, []))

                plot_data = [base_scores, aug_scores, synth_scores]
                labels = ['Baseline', 'Augmented', 'Synthetic']
                
                # plot
                ax.boxplot(plot_data, widths=0.6, showfliers=False, patch_artist=True,
                            medianprops=dict(color="blue", linewidth=1.5),
                            boxprops=dict(edgecolor='black', linewidth=1.5, facecolor='none'),
                            whiskerprops=dict(linestyle='--', color='black', linewidth=1.5),
                            capprops=dict(color='black', linewidth=1.5))

                # plot colored dots for each subject
                for condition_idx, scores in enumerate(plot_data):
                    for subject_idx, score in enumerate(scores):
                        subject_id = all_subjects[subject_idx]
                        color = subject_color_map.get(subject_id)
                        jitter_x = np.random.normal(condition_idx + 1, 0.04)
                        ax.scatter(jitter_x, score, facecolors=color, edgecolors='none',
                                s=65, alpha=0.7, zorder=10)
                
                # plot mean lines
                for condition_idx, scores in enumerate(plot_data):
                    mean_val = np.mean(scores)
                    ax.plot([condition_idx + 1 - 0.3, condition_idx + 1 + 0.3], [mean_val, mean_val],
                            color='red', linestyle='-', linewidth=1.5)
                    # display mean as percentage
                    ax.text(condition_idx + 1 + 0.32, mean_val, f'{mean_val * 100:.1f}%',
                            color='red', va='center', ha='left', fontweight='bold',
                            fontsize=self.font_size - 6)

                # format subplots
                ax.set_title(clf.upper(), fontsize=self.font_size_title)
                ax.set_xticks(np.arange(len(labels)) + 1)
                ax.set_xticklabels(labels, fontsize=self.font_size - 2)
                ax.tick_params(axis='y', labelsize=self.font_size - 2)
                # set y-axis to be percentages
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f'{y * 100:.0f}'))

            # set a single y-label for the whole figure
            axes[0].set_ylabel('Balanced Accuracy (%)', fontsize=self.font_size)
            for ax in axes[1:]:
                ax.tick_params(axis='y', labelleft=True)

            # a single legend for the subjects
            if subject_color_map:
                legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Subject {s}',
                                        markerfacecolor=c, markersize=8, markeredgewidth=0)
                            for s, c in subject_color_map.items()]
                axes[0].legend(handles=legend_elements, title="",
                           loc='lower left',        # position bottom-left
                           bbox_to_anchor=(0.02, 0.05), # small margin from the corner
                           ncol=1,                  # for vertical layout
                           fontsize=self.font_size - 6,
                           frameon=True, framealpha=0.7)
            self.save_figure(fig, experiment_name, f"performance_distribution_{gen_type}")



    def visualize_loso_subject_performance(self, results_data: dict, experiment_name: str):
        """
        Creates figure for each classifier, showing per-subject performance.
        Each figure contains side-by-side subplots for each generator type.
        """
        if results_data['experiment_info']['type'] != 'loso':
            print("Skipping detailed subject plot; this visualization is for LOSO results.")
            return

        subject_perf = results_data.get('subject_performance')
        if not subject_perf:
            print("Warning: 'subject_performance' data not found. Skipping detailed plot.")
            return

        classifiers = self.config.evaluation.classifiers
        generator_types = self.config.generator.generator_types
        n_generators = len(generator_types)
        # subjects are sorted numerically for consistent plotting order
        subjects = sorted(list(subject_perf.get('baseline', {}).keys()), key=int)
        if not subjects:
            print("Warning: No subject data found to visualize.")
            return

        print("--- Visualizing Detailed Subject Performance by Generator ---")

        for clf_name in classifiers:
            fig, axes = plt.subplots(1, n_generators, figsize=(10 * n_generators, 7), sharey=True)
            if n_generators == 1:
                axes = [axes]

            #fig.suptitle(f'{clf_name.upper()} Performance by Subject', fontsize=self.font_size_title, y=0.98)

            # create one subplot per generator
            for i, (ax, gen_type) in enumerate(zip(axes, generator_types)):
                no_aug_values = [subject_perf['baseline'][subj][clf_name] for subj in subjects]
                aug_values = [subject_perf[gen_type][subj][clf_name] for subj in subjects]
                synth_fold_scores = results_data['per_generator_results'][gen_type]['fold_scores']['synthetic'][clf_name]
                subject_indices = [int(s) - 1 for s in subjects]
                synth_values = [synth_fold_scores[idx] for idx in subject_indices]
                
                x = np.arange(len(subjects))
                width = 0.3

                # color palette for bars
                ax.bar(x - width, no_aug_values, width, label='Baseline', color='#8d9396', alpha=0.8)
                ax.bar(x, aug_values, width, label='Augmented', color='#1b85cf', alpha=0.9) # Muted Blue
                ax.bar(x + width, synth_values, width, label='Synthetic-Only', color='#074f82', alpha=0.9) # Muted Orange

                improvements_aug = np.array(aug_values) - np.array(no_aug_values)
                improvements_synth = np.array(synth_values) - np.array(no_aug_values)

                # color palette for average lines
                avg_no_aug = np.mean(no_aug_values)
                avg_aug = np.mean(aug_values)
                avg_synth = np.mean(synth_values) 
                ax.axhline(y=avg_no_aug, color='#8d9396', linestyle='--', alpha=0.7, label=f'Avg Baseline: {avg_no_aug*100:.1f}%')
                ax.axhline(y=avg_aug, color='#1b85cf', linestyle='--', alpha=0.7, label=f'Avg Augmented: {avg_aug*100:.1f}%')
                ax.axhline(y=avg_synth, color='#074f82', linestyle='--', alpha=0.7, label=f'Avg Synthetic: {avg_synth*100:.1f}%')

                for j in range(len(subjects)):
                    # for augmented bar
                    imp_aug = improvements_aug[j]
                    color_aug = 'green' if imp_aug >= 0 else 'red'
                    ax.text(j, aug_values[j] + 0.01, f"{imp_aug*100:+.1f}%",
                            ha='center', va='bottom', color=color_aug,
                            fontsize=self.font_size - 6, fontweight='bold', rotation=90)

                    # for Synthetic-Only bar
                    imp_synth = improvements_synth[j]
                    color_synth = 'green' if imp_synth >= 0 else 'red'
                    ax.text(j + width, synth_values[j] + 0.01, f"{imp_synth*100:+.1f}%",
                            ha='center', va='bottom', color=color_synth,
                            fontsize=self.font_size - 6, fontweight='bold', rotation=90)
                        
                # format
                ax.set_title(f'{gen_type.capitalize()} Generator', fontsize=self.font_size_title - 1)
                ax.set_xlabel('Subject ID', fontsize=self.font_size)
                if ax == axes[0]:
                    ax.set_ylabel('Balanced Accuracy (%)', fontsize=self.font_size)
                ax.grid(axis='y', alpha=0.4, linestyle=':')
                
                # show legend on the last subplot only
                if i == n_generators - 1:
                    ax.legend(fontsize=self.font_size - 2)

                ax.set_xticks(x)
                ax.set_xticklabels([f"{s}" for s in subjects])
                ax.set_ylim(0.2, 1.0)

                all_handles, all_labels = ax.get_legend_handles_labels()

                if i == 0:
                    # the first 3 handles are the bars, the next 3 are the lines
                    line_handles = all_handles[:3]
                    line_labels = all_labels[:3]
                    ax.legend(handles=line_handles, labels=line_labels, loc='upper left', fontsize=self.font_size - 5)

            axes[-1].legend(fontsize=self.font_size - 5, loc='upper right')


            percent_formatter = FuncFormatter(lambda y, pos: f'{y * 100:.0f}')
            axes[0].yaxis.set_major_formatter(percent_formatter)
            
            for ax in axes[1:]: # loop through all axes except the first one
                ax.tick_params(axis='y', labelleft=True)

            fig.subplots_adjust(wspace=0.12)
            self.save_figure(fig, experiment_name, f"subject_perf_{clf_name}_{gen_type}")

    def visualize_within_subject_performance(self, results_data: dict, experiment_name: str, subject_id: int):
        """Creates a bar chart for a single subject's train/test evaluation"""
        print(f"--- Visualizing Within-Subject Performance for Subject {subject_id} ---")
        baseline_scores = results_data.get('baseline_results', {}).get('average_metrics', {})
        if not baseline_scores:
            print("Warning: Baseline results not found. Cannot generate within-subject plot.")
            return

        classifiers = list(baseline_scores.keys())
        x = np.arange(len(classifiers))
        width = 0.25

        # generate a plot for each generator type
        for gen_type in self.config.generator.generator_types:
            gen_results = results_data.get('per_generator_results', {}).get(gen_type, {})
            if not gen_results:
                continue

            aug_scores = gen_results.get('average_metrics', {}).get('aug', {})
            synth_scores = gen_results.get('average_metrics', {}).get('synthetic', {})

            fig, ax = plt.subplots(figsize=(12, 7))

            bar1 = ax.bar(x - width, baseline_scores.values(), width, label='Baseline', color='steelblue')
            bar2 = ax.bar(x, aug_scores.values(), width, label='Augmented', color='forestgreen')
            bar3 = ax.bar(x + width, synth_scores.values(), width, label='Synthetic-Only', color='darkorange')

            ax.set_ylabel('Balanced Accuracy', fontsize=self.font_size)
            ax.set_title(f'Within-Subject Performance for S{subject_id} (Generator: {gen_type.capitalize()})', fontsize=self.font_size_title)
            ax.set_xticks(x)
            ax.set_xticklabels([clf.upper() for clf in classifiers], fontsize=self.font_size)
            ax.legend(fontsize=self.font_size - 2)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.set_ylim(0, 1.05)

            ax.bar_label(bar1, fmt='%.3f', padding=3, fontsize=self.font_size - 4)
            ax.bar_label(bar2, fmt='%.3f', padding=3, fontsize=self.font_size - 4)
            ax.bar_label(bar3, fmt='%.3f', padding=3, fontsize=self.font_size - 4)

            self.save_figure(fig, f"{experiment_name}_s{subject_id}", f"_performance_{gen_type}")

    def visualize_matrix_examples(self, model, real_data, device, n_examples=2, name_prefix="matrix_examples", fold_id=""):
        """
        Visualizes real matrices, their reconstructions, and novel synthetic matrices
        in a grid with examples as rows and conditions as columns.

        Args:
            model (torch.nn.Module): The trained VAE model
            real_data (np.ndarray): A sample of real covariance matrices
            device (torch.device): The device
            n_examples (int): The number of examples to show (number of rows)
            name_prefix (str): A prefix for the saved figure file
            fold_id (str): An identifier for the fold being visualized
        """
        model.eval()
        
        # get real and reconstructed data
        indices = np.random.choice(len(real_data), n_examples, replace=False)
        X_real = torch.tensor(real_data[indices], dtype=torch.float32).to(device)
        with torch.no_grad():
            X_recon, _, _, _, _, _ = model(X_real)
        
        # get novel synthetic data (from prior)
        with torch.no_grad():
            z_prior = torch.randn(n_examples, model.latent_dim, device=device)
            X_synth, _ = model.decode(z_prior)

        fig, axes = plt.subplots(n_examples, 3, figsize=(12, 4 * n_examples))

        title = "Comparison of Covariance Matrices"
        #if fold_id:
        #    title += f" ({fold_id})"
        fig.suptitle(title, fontsize=self.font_size_title)
        
        # calculate a single, symmetric color limit across ALL matrices for consistency
        all_matrices = torch.cat([X_real, X_recon, X_synth]).cpu().numpy()
        vmax_abs = np.abs(all_matrices).max()

        column_titles = ['(a) Real Input', '(b) VAE Reconstruction', '(c) Novel Synthetic']

        for i in range(n_examples): # loop through rows, where each row is an example
            # data for the current row
            real_matrix = X_real[i].cpu().numpy()
            recon_matrix = X_recon[i].cpu().numpy()
            synth_matrix = X_synth[i].cpu().numpy()

            # Column 0: Real Matrix
            axes[i, 0].imshow(real_matrix, cmap='coolwarm', interpolation='nearest',
                            vmin=-vmax_abs, vmax=vmax_abs)
            
            # Column 1: Reconstructed Matrix
            axes[i, 1].imshow(recon_matrix, cmap='coolwarm', interpolation='nearest',
                            vmin=-vmax_abs, vmax=vmax_abs)

            # Column 2: Synthetic Matrix
            im = axes[i, 2].imshow(synth_matrix, cmap='coolwarm', interpolation='nearest',
                                vmin=-vmax_abs, vmax=vmax_abs)

            # formatting for the row
            axes[i, 0].set_ylabel(f'Example {i+1}', fontsize=self.font_size_title)
            for j in range(3):
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])

        # set titles for the columns on the top row
        for j in range(3):
            axes[0, j].set_title(column_titles[j], fontsize=self.font_size_title)

        # add a single, shared colorbar for the entire figure
        fig.subplots_adjust(right=0.85, hspace=0.1, wspace=0.1)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Covariance Value', fontsize=self.font_size)
        
        self.save_figure(fig, name_prefix)


    def visualize_spd_properties(self, results_data, experiment_name, result_type='loso'):
        pass

    def visualize_loso_generalization_space(self, model_to_encode_with,
                                        X_train, subjects_train,
                                        X_test, subjects_test,
                                        synthetic_data,
                                        device, experiment_name, fold_id):
        """
        For a single LOSO fold, visualizes how synthetic data and unseen test data
        map to the learned latent space of the training data.

        Args:
            model_to_encode_with (torch.nn.Module): The trained model for this fold
            X_train (np.ndarray): The aligned training covariance matrices
            subjects_train (np.ndarray): The subject IDs for the training data
            X_test (np.ndarray): The covariance matrices for the left-out test subject
            subjects_test (np.ndarray): The subject ID for the test data
            synthetic_data (np.ndarray): The generated synthetic covariance matrices
            device (torch.device): The device to use for tensor operations
            experiment_name (str): The base name for the experiment
            fold_id (str): An identifier for the current fold (e.g 'test_subject_1')
        """
        print(f"\n--- Generating LOSO Generalization Space visualization for fold: {fold_id} ---")
        model_to_encode_with.eval()

        def encode_data(X_data):
            if len(X_data) == 0: return np.array([])
            with torch.no_grad():
                mu, _, _ = model_to_encode_with.encode(torch.tensor(X_data, dtype=torch.float32).to(device))
            return mu.cpu().numpy()

        # UMAP Fitting: fit ONLY on the training data to create a reference projection
        train_latent_vecs = encode_data(X_train)
        if len(train_latent_vecs) < 15: # UMAP needs a minimum number of neighbors
            print(f"Not enough training samples ({len(train_latent_vecs)}) for robust UMAP.")
            return

        umap_reducer = UMAP(n_components=2, random_state=42, n_neighbors=15)
        reduced_train_vecs = umap_reducer.fit_transform(train_latent_vecs)

        # transform other datasets using the FITTED reducer
        synthetic_latent_vecs = encode_data(synthetic_data)
        reduced_synthetic_vecs = umap_reducer.transform(synthetic_latent_vecs)

        test_latent_vecs = encode_data(X_test)
        reduced_test_vecs = umap_reducer.transform(test_latent_vecs)

        fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharex=True, sharey=True)
        #fig.suptitle(f'Generalization Analysis in Latent Space for {fold_id}', fontsize=self.font_size_title, y=1.0)

        unique_train_subjects = np.unique(subjects_train)
        palette = sns.color_palette("husl", n_colors=len(unique_train_subjects))
        subject_color_map = {subject: palette[i] for i, subject in enumerate(unique_train_subjects)}
        test_subject_id = subjects_test[0]

        # plot 1
        ax1 = axes[0]
        sns.scatterplot(x=reduced_train_vecs[:, 0], y=reduced_train_vecs[:, 1], hue=subjects_train,
                        palette=subject_color_map, ax=ax1, s=30, alpha=0.8)
        ax1.set_title('(a) Learned Training Manifold', fontsize=self.font_size_title)
        ax1.legend(title='Train Subject')

        # plot 2
        ax2 = axes[1]
        sns.scatterplot(x=reduced_train_vecs[:, 0], y=reduced_train_vecs[:, 1], hue=subjects_train,
                        palette=subject_color_map, ax=ax2, s=30, alpha=0.1) # fade train data
        ax2.scatter(reduced_synthetic_vecs[:, 0], reduced_synthetic_vecs[:, 1], c='black',
                    marker='x', s=35, label='Synthetic', alpha=0.5)
        ax2.set_title('(b) Synthetic Data in Manifold', fontsize=self.font_size_title)
        ax2.legend(title='Data Type')

        # plot 3
        ax3 = axes[2]
        sns.scatterplot(x=reduced_train_vecs[:, 0], y=reduced_train_vecs[:, 1], hue=subjects_train,
                        palette=subject_color_map, ax=ax3, s=30, alpha=0.1) 
        ax3.scatter(reduced_test_vecs[:, 0], reduced_test_vecs[:, 1], c='magenta',
                    marker='*', s=100, label=f'Unseen (S{test_subject_id})', edgecolors='black')
        ax3.set_title('(c) Unseen Subject in Manifold', fontsize=self.font_size_title)
        ax3.legend(title='Data Type')

        for ax in axes:
            ax.set_xlabel('UMAP Component 1', fontsize=self.font_size)
            ax.grid(True, linestyle=':', alpha=0.6)
        axes[0].set_ylabel('UMAP Component 2', fontsize=self.font_size)

        self.save_figure(fig, experiment_name, f"loso_generalization_space_{fold_id}")
        
    def visualize_latent_space_stages(self, model_to_encode_with, data_dict, device, experiment_name):
        """
        Visualizes latent space for original, transported, and synthetic data.
        A single UMAP model is fitted on the aligned data and used to transform all
        stages for a consistent and accurate comparison.
        """
        model_to_encode_with.eval()

        # define stages and titles text
        stages = ['original_unaligned_cov', 'original_aligned_cov', 'synthetic_cov']
        plot_titles = {
            'original_unaligned_cov': '(a) Original (Unaligned) Latent Space',
            'original_aligned_cov': '(b) Aligned Latent Space',
            'synthetic_cov': '(c) Synthetic vs. Aligned Latent Space'
        }
        
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        #fig.suptitle('UMAP of Data Distribution at Different Pipeline Stages', fontsize=self.font_size_title, y=0.9)

        # get all unique subjects for a consistent color palette
        all_subjects = np.unique(np.concatenate([v['subjects'] for k, v in data_dict.items() if 'subjects' in v and k != 'synthetic_cov']))
        palette = sns.color_palette("husl", n_colors=len(all_subjects))
        subject_color_map = {subject: palette[i] for i, subject in enumerate(all_subjects)}
        
        def encode_data(X_data): # TODO: make into class helper function
            if len(X_data) == 0: return np.array([])
            with torch.no_grad():
                mu, _, _ = model_to_encode_with.encode(torch.tensor(X_data, dtype=torch.float32).to(device))
            return mu.cpu().numpy()

        # fit UMAP only ONCE on the aligned data
        aligned_latent_vectors = encode_data(data_dict['original_aligned_cov']['X'])
        umap_reducer = UMAP(n_components=2, random_state=11, n_neighbors=15, min_dist=0.1)
        if len(aligned_latent_vectors) > 0:
            umap_reducer.fit(aligned_latent_vectors)

        # process and plot each stage
        for i, stage in enumerate(stages):
            ax = axes[i]
            
            if stage == 'synthetic_cov':
                # for the final plot, transform both datasets using the already-fitted reducer
                aligned_data = data_dict['original_aligned_cov']
                synth_data = data_dict['synthetic_cov']
                
                latent_synth = encode_data(synth_data['X'])
                
                if len(aligned_latent_vectors) > 0:
                    reduced_aligned = umap_reducer.transform(aligned_latent_vectors)
                    # plot real data colored by subject
                    for subject in all_subjects:
                        mask = (aligned_data['subjects'] == subject)
                        ax.scatter(reduced_aligned[mask, 0], reduced_aligned[mask, 1], c=[subject_color_map.get(subject)], label=f'Subject {subject}', s=38, alpha=0.9)

                if len(latent_synth) > 0:
                    reduced_synth = umap_reducer.transform(latent_synth)
                    # plot synthetic data with transparent x markers
                    ax.scatter(reduced_synth[:, 0], reduced_synth[:, 1], c='black', marker='x', s=35, label='Synthetic', alpha=0.6)

            else: # for stage (a) and (b)
                stage_data = data_dict[stage]
                latent_vecs = encode_data(stage_data['X'])
                
                if len(latent_vecs) > 0:
                    # for plot (b), use the fitted reducer. for (a), fit a new one.
                    if stage == 'original_aligned_cov':
                        reduced_vecs = umap_reducer.transform(latent_vecs)
                    else:
                        # unaligned data gets its own UMAP projection
                        temp_reducer = UMAP(n_components=2, random_state=42)
                        reduced_vecs = temp_reducer.fit_transform(latent_vecs)
                    
                    for subject in all_subjects:
                        mask = (stage_data['subjects'] == subject)
                        ax.scatter(reduced_vecs[mask, 0], reduced_vecs[mask, 1], c=[subject_color_map.get(subject)], label=f'Subject {subject}', s=30, alpha=0.9)
            
            ax.set_title(plot_titles[stage], fontsize=self.font_size_title)
            ax.set_xlabel('UMAP Component 1', fontsize=self.font_size)
            if i == 0:
                ax.set_ylabel('UMAP Component 2', fontsize=self.font_size)
            else:
                ax.set_ylabel('') # clear y-label for next subplots
            if stage == 'synthetic_cov':
                ax.legend(fontsize=self.font_size - 4)
            ax.grid(True, linestyle=':', alpha=0.6)

        fig.tight_layout(rect=[0, 0.03, 1, 0.93])
        self.save_figure(fig, experiment_name, "_latent_space_stages_comparison")


    def visualize_improvement_vs_baseline(self, results_data: dict, experiment_name: str):
        """
        Creates a grid of scatter plots to analyze the relationship between a
        subject's baseline performance and the improvement gained from augmentation.

        Generates one figure per generator type, with two rows of plots:
        1. Top Row: improvement from Augmented (Real+Synth) data vs. Baseline.
        2. Bottom Row: improvement from Synthetic-Only data vs. Baseline.
        """
        if results_data['experiment_info']['type'] != 'loso':
            return

        baseline_results = results_data.get('baseline_results', {})
        classifiers = list(baseline_results.get('average_metrics', {}).keys())
        if not classifiers:
            return

        # draw a single scatter plo
        def _plot_scatter_subplot(ax, base_scores, improvements, title):
            ax.scatter(base_scores, improvements, alpha=0.8, s=60, edgecolors='k', zorder=10)

            # fit and plot a line of best fit
            if len(base_scores) > 1:
                m, b = np.polyfit(base_scores, improvements, 1)
                ax.plot(np.array(base_scores), m * np.array(base_scores) + b, color='red', linestyle='--', linewidth=2)

                # calculate and display correlation
                corr, p_val = pearsonr(base_scores, improvements)
                ax.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}',
                        transform=ax.transAxes, fontsize=self.font_size - 2,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.axhline(0, color='black', linestyle=':', alpha=0.7)
            ax.set_title(title, fontsize=self.font_size_title)
            ax.grid(True, linestyle=':', alpha=0.5)

        print("--- Visualizing Correlation between Baseline Performance and Improvement ---")

        for gen_type in self.config.generator.generator_types:
            # 2-row figure for each generator type
            fig, axes = plt.subplots(2, len(classifiers), figsize=(8 * len(classifiers), 11), sharex=True, sharey=True)
            if len(classifiers) == 1:
                axes = axes.reshape(2, 1) # ensure axes always 2D

            fig.suptitle(f'Improvement vs. Baseline Performance (Generator: {gen_type.capitalize()})',
                        fontsize=self.font_size_title, y=0.95)

            for i, clf in enumerate(classifiers):
                base_scores = baseline_results.get('fold_scores', {}).get(clf, [])
                if not base_scores:
                    continue

                # plot for top row (Augmented)
                ax_aug = axes[0, i]
                aug_scores = results_data['per_generator_results'][gen_type].get('fold_scores', {}).get('aug', {}).get(clf, [])
                if aug_scores:
                    improvements_aug = np.array(aug_scores) - np.array(base_scores)
                    _plot_scatter_subplot(ax_aug, base_scores, improvements_aug, f'{clf.upper()}')

                # plot for bottom row (Synthetic-Only)
                ax_synth = axes[1, i]
                synth_scores = results_data['per_generator_results'][gen_type].get('fold_scores', {}).get('synthetic', {}).get(clf, [])
                if synth_scores:
                    improvements_synth = np.array(synth_scores) - np.array(base_scores)
                    _plot_scatter_subplot(ax_synth, base_scores, improvements_synth, f'{clf.upper()}')

                # set axis labels
                ax_synth.set_xlabel('Baseline Accuracy', fontsize=self.font_size)
                if i == 0:
                    ax_aug.set_ylabel('Improvement (Augmented)', fontsize=self.font_size)
                    ax_synth.set_ylabel('Improvement (Synthetic-Only)', fontsize=self.font_size)

            self.save_figure(fig, experiment_name, f"improvement_vs_baseline_{gen_type}")


class EegVisualizer(BaseVisualizer):
    def plot_eeg_transformations(self, raw_epoch_data, standardized_epoch_data, cov_matrix,
                             sampling_rate=250, name_prefix="eeg_transformations", trial_id="N/A"):
        """
        Plots and saves key EEG preprocessing steps as three separate figures for a single trial.
        
        Args:
            raw_epoch_data (np.ndarray): EEG data after scaling to microvolts
            standardized_epoch_data (np.ndarray): Data after Standardization
            cov_matrix (np.ndarray): The final covariance matrix
            sampling_rate (float): The sampling rate of the EEG data
            name_prefix (str): A prefix for the saved figure files
            trial_id (str): An identifier for the trial being plotted
        """
        if not all(isinstance(arr, np.ndarray) and arr.ndim == 2 for arr in [raw_epoch_data, standardized_epoch_data, cov_matrix]):
            print("Error: Input data must be 2D NumPy arrays.")
            raise
            
        n_channels, n_times = raw_epoch_data.shape
        time_axis = np.arange(n_times) / sampling_rate
        safe_trial_id = trial_id.replace(' ', '_').replace('/', '-')

        # plot 1
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        vmax_raw = np.abs(raw_epoch_data).max()
        im1 = ax1.imshow(raw_epoch_data, aspect='auto', cmap='coolwarm',
                            vmin=-vmax_raw, vmax=vmax_raw,
                            extent=[time_axis[0], time_axis[-1], n_channels, 0])
        ax1.set_title('(a) Scaled Signal', fontsize=self.font_size_title)
        ax1.set_xlabel('Time (s)', fontsize=self.font_size)
        ax1.set_ylabel('Channel Index', fontsize=self.font_size)
        ax1.tick_params(axis='both', which='major', labelsize=self.font_size - 2)
        cbar1 = fig1.colorbar(im1, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)
        cbar1.set_label('Amplitude (μV)', fontsize=self.font_size)
        cbar1.ax.tick_params(labelsize=self.font_size - 2)
        self.save_figure(fig1, name_prefix, f"{safe_trial_id}_a_scaled_signal")

        # plot 2
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        vmax_std = np.abs(standardized_epoch_data).max()
        im2 = ax2.imshow(standardized_epoch_data, aspect='auto', cmap='coolwarm',
                            vmin=-vmax_std, vmax=vmax_std,
                            extent=[time_axis[0], time_axis[-1], n_channels, 0])
        ax2.set_title('(b) Standardized Signal', fontsize=self.font_size_title)
        ax2.set_xlabel('Time (s)', fontsize=self.font_size)
        ax2.set_ylabel('Channel Index', fontsize=self.font_size)
        ax2.tick_params(axis='both', which='major', labelsize=self.font_size - 2)
        cbar2 = fig2.colorbar(im2, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04)
        cbar2.set_label('Amplitude (Standardized)', fontsize=self.font_size)
        cbar2.ax.tick_params(labelsize=self.font_size - 2)
        self.save_figure(fig2, name_prefix, f"{safe_trial_id}_b_standardized_signal")

        # plot 3
        fig3, ax3 = plt.subplots(figsize=(5, 5))
        
        # set symmetric color limits for the coolwarm map
        vmax_cov = np.abs(cov_matrix).max()
        im3 = ax3.imshow(cov_matrix, cmap='coolwarm', interpolation='nearest',
                            vmin=-vmax_cov, vmax=vmax_cov)
        ax3.set_title('(c) Covariance Matrix', fontsize=self.font_size_title)
        ax3.set_xlabel('Channel Index', fontsize=self.font_size)
        ax3.set_ylabel('Channel Index', fontsize=self.font_size)
        ax3.tick_params(axis='both', which='major', labelsize=self.font_size - 2)
        cbar3 = fig3.colorbar(im3, ax=ax3)
        cbar3.set_label('Covariance', fontsize=self.font_size)
        cbar3.ax.tick_params(labelsize=self.font_size - 2)
        self.save_figure(fig3, name_prefix, f"{safe_trial_id}_c_covariance_matrix")
