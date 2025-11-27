"""
Run the IGANEEG ablation experiments in batches.
Train and test 4 model variants on 4 data configurations.
"""




sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ablation_models import ABLATION_CONFIGS, ABLATION_NAMES
from ablation_training import run_single_ablation
from ablation_testing import run_single_ablation_test

class AblationExperimentRunner:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.results_dir = os.path.join(self.base_dir, 'ablation_results')
        
     
        os.makedirs(self.results_dir, exist_ok=True)
        
      
        self.ablation_types = [1, 2, 3, 4] 
        self.data_configs = [
            "win64_overlap0",
            "win64_overlap32", 
            "win128_overlap0",
            "win128_overlap32"
        ]  
        
        self.results = {}
        self.experiment_start_time = None
        
    def run_single_experiment(self, ablation_type, data_config):
     
        print(f"\n{'='*100}")
        print(f"Start Ablation study: {ABLATION_NAMES[ABLATION_CONFIGS[ablation_type]]} - {data_config}")
        print(f"{'='*100}")
        
        start_time = time.time()
        
        try:
          
            print(f"\n🚀 Start the training....")
            train_success = run_single_ablation(ablation_type, data_config)
            
            if not train_success:
                print(f"❌ The training has failed: {ABLATION_NAMES[ABLATION_CONFIGS[ablation_type]]} - {data_config}")
                return None
            
           
            print(f"\n🧪 Start the testing...")
            test_results = run_single_ablation_test(ablation_type, data_config)
            
            if test_results is None:
                print(f"❌ . The testing has failed: {ABLATION_NAMES[ABLATION_CONFIGS[ablation_type]]} - {data_config}")
                return None
            
            end_time = time.time()
            duration = end_time - start_time
            
           
            experiment_result = {
                'ablation_type': ablation_type,
                'ablation_name': ABLATION_NAMES[ABLATION_CONFIGS[ablation_type]],
                'data_config': data_config,
                'train_success': True,
                'test_success': True,
                'duration': duration,
                'start_time': datetime.fromtimestamp(start_time),
                'end_time': datetime.fromtimestamp(end_time),
                'metrics': {
                    'accuracy': test_results['accuracy'],
                    'mae': test_results['mae'],
                    'mse': test_results['mse'],
                    'mmd': test_results['mmd']
                }
            }
            
            print(f"✅ The experiment is completed: {ABLATION_NAMES[ABLATION_CONFIGS[ablation_type]]} - {data_config}")
            print(f"⏱️ Time taken: {duration:.2f} s")
            print(f"📊 Accuracy: {test_results['accuracy']:.4f}")
            print(f"📊 MAE: {test_results['mae']:.4f}")
            print(f"📊 MSE: {test_results['mse']:.4f}")
            print(f"📊 MMD: {test_results['mmd']:.6f}")
            
            return experiment_result
            
        except Exception as e:
            print(f"❌  Errors occurred during the experiment: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_all_experiments(self, mode='sequential'):

        print("🧠 IGANEEG Ablation Experiment Batch Running Tool")
        print("=" * 100)
        print(f"Experiment mode: {mode}")
        print(f"Ablation type: {', '.join([ABLATION_NAMES[ABLATION_CONFIGS[t]] for t in self.ablation_types])}")
        print(f"Data configuration: {', '.join(self.data_configs)}")
        print(f"Total number of experiments: {len(self.ablation_types) * len(self.data_configs)}")
        print("=" * 100)
        
        self.experiment_start_time = time.time()
        total_experiments = len(self.ablation_types) * len(self.data_configs)
        completed_experiments = 0
        
        if mode == 'sequential':
          
            for ablation_type in self.ablation_types:
                for data_config in self.data_configs:
                    result = self.run_single_experiment(ablation_type, data_config)
                    
                    if result:
                        key = f"{ABLATION_CONFIGS[ablation_type]}_{data_config}"
                        self.results[key] = result
                        completed_experiments += 1
                    
                    print(f"\n📈 progress: {completed_experiments}/{total_experiments} ({completed_experiments/total_experiments*100:.1f}%)")
        
        elif mode == 'parallel':
            print(  "⚠Warning: Running in parallel may consume a large amount of resources. It is recommended to use sequential running.")
            self.run_all_experiments('sequential')
        
        else:
            raise ValueError(f"Unknown running mode: {mode}")
        
        total_end_time = time.time()
        total_duration = total_end_time - self.experiment_start_time
        
        print(f"\n{'='*100}")
        print("🎉 All ablation experiments are completed!")
        print(f"⏱️  Total time taken: {total_duration:.2f} seconds")
        print(f"✅ Success: {completed_experiments}/{total_experiments} experiments")
        print(f"{'='*100}")
        
        
        self.analyze_results()
        self.save_results()
        self.generate_report()
        
        return self.results
    
    def analyze_results(self):

        print("\n📊 Analyzing the experimental results...")

        if not self.results:
            print("❌ There are no results available for analysis.")
            return
        
 
        ablation_results = {}
        for key, result in self.results.items():
            ablation_type = result['ablation_type']
            if ablation_type not in ablation_results:
                ablation_results[ablation_type] = []
            ablation_results[ablation_type].append(result)
        
       
        self.ablation_summary = {}
        
        for ablation_type, results in ablation_results.items():
            accuracies = [r['metrics']['accuracy'] for r in results]
            maes = [r['metrics']['mae'] for r in results]
            mses = [r['metrics']['mse'] for r in results]
            mmds = [r['metrics']['mmd'] for r in results]
            
            self.ablation_summary[ablation_type] = {
                'name': ABLATION_NAMES[ABLATION_CONFIGS[ablation_type]],
                'count': len(results),
                'accuracy_mean': np.mean(accuracies),
                'accuracy_std': np.std(accuracies),
                'mae_mean': np.mean(maes),
                'mae_std': np.std(maes),
                'mse_mean': np.mean(mses),
                'mse_std': np.std(mses),
                'mmd_mean': np.mean(mmds),
                'mmd_std': np.std(mmds)
            }

        print(f"\n📋 Summary of ablation experiment results:")
        print("-" * 100)
        print(f"{'Ablation Type':<25} {'Quantity':<6} {'Accuracy':<12} {'MAE':<12} {'MSE':<12} {'MMD':<12}")
        print("-" * 100)
        
        for ablation_type, summary in self.ablation_summary.items():
            print(f"{summary['name']:<25} {summary['count']:<6} "
                  f"{summary['accuracy_mean']:.4f}±{summary['accuracy_std']:.4f}  "
                  f"{summary['mae_mean']:.4f}±{summary['mae_std']:.4f}  "
                  f"{summary['mse_mean']:.4f}±{summary['mse_std']:.4f}  "
                  f"{summary['mmd_mean']:.6f}±{summary['mmd_std']:.6f}")
    
    def save_results(self):
     
        print(f"\n💾 save results...")
        
       
        results_file = os.path.join(self.results_dir, 'detailed_results.pkl')
        with open(results_file, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"The detailed results have been saved to : {results_file}")
        
    
        if hasattr(self, 'ablation_summary'):
            summary_file = os.path.join(self.results_dir, 'ablation_summary.pkl')
            with open(summary_file, 'wb') as f:
                pickle.dump(self.ablation_summary, f)
            print(f"the summary results have been saved to : {summary_file}")
            
         
            csv_file = os.path.join(self.results_dir, 'ablation_summary.csv')
            df_data = []
            for ablation_type, summary in self.ablation_summary.items():
                df_data.append({
                    'Ablation_Type': summary['name'],
                    'Count': summary['count'],
                    'Accuracy_Mean': summary['accuracy_mean'],
                    'Accuracy_Std': summary['accuracy_std'],
                    'MAE_Mean': summary['mae_mean'],
                    'MAE_Std': summary['mae_std'],
                    'MSE_Mean': summary['mse_mean'],
                    'MSE_Std': summary['mse_std'],
                    'MMD_Mean': summary['mmd_mean'],
                    'MMD_Std': summary['mmd_std']
                })
            
            df = pd.DataFrame(df_data)
            df.to_csv(csv_file, index=False)
            print(f"The CSV summary has been saved to: {csv_file}")
    
    def generate_report(self):
     
        print(f"\n📄 Generate an experiment report...")
        
        if not hasattr(self, 'ablation_summary'):
            print("❌ there is no data available for report generation")
            return
        
       
        self.plot_results()
        
    
        report_file = os.path.join(self.results_dir, 'experiment_report.txt')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            ```python
            f.write("IGANEEG Ablation Experiment Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total number of experiments: {len(self.results)}\n")
            f.write(f"Number of ablation types: {len(self.ablation_types)}\n")
            f.write(f"Number of data configurations: {len(self.data_configs)}\n\n")

            f.write("Summary of experiment results:\n")

            ```
            f.write("-" * 50 + "\n")
            
            for ablation_type, summary in self.ablation_summary.items():
                f.write(f"\n{summary['name']}:\n")
                f.write(f"  Number: {summary['count']}\n")
                f.write(f"  Accuracy: {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}\n")
                f.write(f"  MAE: {summary['mae_mean']:.4f} ± {summary['mae_std']:.4f}\n")
                f.write(f"  MSE: {summary['mse_mean']:.4f} ± {summary['mse_std']:.4f}\n")
                f.write(f"  MMD: {summary['mmd_mean']:.6f} ± {summary['mmd_std']:.6f}\n")
            
          
            best_accuracy = max(self.ablation_summary.items(), 
                              key=lambda x: x[1]['accuracy_mean'])
            best_mae = min(self.ablation_summary.items(), 
                         key=lambda x: x[1]['mae_mean'])
            best_mse = min(self.ablation_summary.items(), 
                         key=lambda x: x[1]['mse_mean'])
            best_mmd = min(self.ablation_summary.items(), 
                         key=lambda x: x[1]['mmd_mean'])

            f.write(f"\nBest configuration:\n")
            f.write(f"  Highest Accuracy: {best_accuracy[1]['name']} ({best_accuracy[1]['accuracy_mean']:.4f})\n")
            f.write(f"  Lowest MAE: {best_mae[1]['name']} ({best_mae[1]['mae_mean']:.4f})\n")
            f.write(f"  Lowest MSE: {best_mse[1]['name']} ({best_mse[1]['mse_mean']:.4f})\n")
            f.write(f"  Lowest MMD: {best_mmd[1]['name']} ({best_mmd[1]['mmd_mean']:.6f})\n")
        
        print(f"The experiment report has been saved to: {report_file}")
    
    def plot_results(self):
    
        if not hasattr(self, 'ablation_summary'):
            return
        
       
        names = [summary['name'] for summary in self.ablation_summary.values()]
        accuracies = [summary['accuracy_mean'] for summary in self.ablation_summary.values()]
        accuracy_stds = [summary['accuracy_std'] for summary in self.ablation_summary.values()]
        maes = [summary['mae_mean'] for summary in self.ablation_summary.values()]
        mae_stds = [summary['mae_std'] for summary in self.ablation_summary.values()]
        mses = [summary['mse_mean'] for summary in self.ablation_summary.values()]
        mse_stds = [summary['mse_std'] for summary in self.ablation_summary.values()]
        
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy
        ax1.bar(names, accuracies, yerr=accuracy_stds, capsize=5, alpha=0.7)
        ax1.set_title('Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # MAE
        ax2.bar(names, maes, yerr=mae_stds, capsize=5, alpha=0.7, color='orange')
        ax2.set_title('MAE Comparison')
        ax2.set_ylabel('MAE')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # MSE
        ax3.bar(names, mses, yerr=mse_stds, capsize=5, alpha=0.7, color='red')
        ax3.set_title('MSE Comparison')
        ax3.set_ylabel('MSE')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
     
        metrics = ['Accuracy', 'MAE', 'MSE']
        x = np.arange(len(names))
        width = 0.25
        
     
        norm_acc = np.array(accuracies)
        norm_mae = -np.array(maes) / np.max(maes)  
        norm_mse = -np.array(mses) / np.max(mses)  
        
        ax4.bar(x - width, norm_acc, width, label='Accuracy', alpha=0.7)
        ax4.bar(x, norm_mae, width, label='MAE (normalized)', alpha=0.7)
        ax4.bar(x + width, norm_mse, width, label='MSE (normalized)', alpha=0.7)
        
        ax4.set_title('Normalized Metrics Comparison')
        ax4.set_ylabel('Normalized Score')
        ax4.set_xticks(x)
        ax4.set_xticklabels(names, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_file = os.path.join(self.results_dir, 'ablation_results.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"The result charts have been saved to: {plot_file}")

def main():
    parser = argparse.ArgumentParser(description='Run IGANEEG ablation experiments in batches')
    parser.add_argument('--mode', choices=['sequential', 'parallel'],
                        default='sequential', help='Running mode')
    parser.add_argument('--ablation-types', nargs='+', type=int,
                        choices=[1, 2, 3, 4], help='Specify ablation types')
    parser.add_argument('--data-configs', nargs='+',
                        choices=['win64_overlap0', 'win64_overlap32', 'win128_overlap0', 'win128_overlap32'],
                        help='Specify data configurations')
    
    args = parser.parse_args()
    
    
    runner = AblationExperimentRunner()
    
    
    if args.ablation_types:
        runner.ablation_types = args.ablation_types
    if args.data_configs:
        runner.data_configs = args.data_configs
    
  
    results = runner.run_all_experiments(args.mode)

    if results:
        print("\n🎉 Ablation experiments completed successfully!")
        print(f"📁 Results are saved in: {runner.results_dir}")
    else:
        print("\n❌ Ablation experiments failed!")

if __name__ == "__main__":
    main()