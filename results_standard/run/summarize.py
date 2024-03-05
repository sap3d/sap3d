import os
import json
from collections import defaultdict

# Define the path to the directory containing the experiment results
base_dir = "results_standard/GSO_demo"

# Function to collect all results into a dictionary grouped by NAME, with each NAME containing a dictionary of views
def collect_and_group_results(base_dir):
    grouped_results = {}
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith("results.json"):  # Only read 'results.json' files
                json_path = os.path.join(root, file)
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    if data and isinstance(data, list):
                        # Assuming the view information is encoded in the folder structure
                        path_parts = root.split(os.sep)
                        view = path_parts[-1]  # Assuming the last part of the path indicates the view
                        for item in data:
                            name = item['NAME']
                            if name not in grouped_results:
                                grouped_results[name] = {}
                            if view not in grouped_results[name]:
                                grouped_results[name][view] = []
                            grouped_results[name][view].append(item)
    return grouped_results

# Calculate the average metrics across all views for all objects
def calculate_average_metrics_across_views(results):
    metrics_of_interest = [
        "CD", "IOU", "F1_01", "F1_02", "F1_03", "Rotation Error", "Elevation",
        "Azimuth", "Radius", "2d_psnr_mean", "2d_ssim_mean", "2d_lpips_mean",
        "3d_psnr_mean", "3d_ssim_mean", "3d_lpips_mean"
    ]
    view_sums = defaultdict(lambda: defaultdict(float))
    view_counts = defaultdict(lambda: defaultdict(int))
    
    for name, views in results.items():
        for view, view_results in views.items():
            for result in view_results:
                for metric in metrics_of_interest:
                    if metric in result:
                        view_sums[view][metric] += result[metric]
                        view_counts[view][metric] += 1
    
    view_averages = defaultdict(dict)
    for view in view_sums:
        for metric in view_sums[view]:
            if view_counts[view][metric] > 0:
                view_averages[view][metric] = view_sums[view][metric] / view_counts[view][metric]
    return view_averages

def print_latex_tables_for_views(view_averages, category_titles):
    for category, metrics in category_titles.items():
        print(f"\\begin{{table}}[H]\n\\centering\n\\begin{{tabular}}{{l|{'c' * len(metrics)}}}\n\\toprule")
        print("View & " + " & ".join([metric.replace('_', '\\_') for metric in metrics]) + " \\\\\n\\midrule")
        sorted_views = sorted(view_averages.keys(), key=lambda x: int(x.split('_')[-1]))
        
        for view in sorted_views:
            row = [view]
            for metric in metrics:
                if metric in view_averages[view]:
                    value = view_averages[view][metric]
                    formatted_value = f"{value:.3f}" if metric == "CD" else f"{value:.2f}"
                    row.append(formatted_value)
                else:
                    row.append("N/A")
            print(" & ".join(row) + " \\\\")
        
        print("\\bottomrule\n\\end{tabular}")
        print(f"\\caption{{Average {category} across all views.}}\n\\label{{tab:{category.lower().replace(' ', '_')}}}\n\\end{{table}}\n\n")

# Main execution
if __name__ == "__main__":
    grouped_results = collect_and_group_results(base_dir)
    view_averages = calculate_average_metrics_across_views(grouped_results)

    category_titles = {
        "2D Metrics": ["2d_psnr_mean", "2d_ssim_mean", "2d_lpips_mean"],
        "3D Metrics": ["3d_psnr_mean", "3d_ssim_mean", "3d_lpips_mean"],
        "3D Mesh Metrics": ["CD", "IOU", "F1_01", "F1_02", "F1_03"],
        "Rotation Error Metrics": ["Rotation Error"]
    }

    print_latex_tables_for_views(view_averages, category_titles)