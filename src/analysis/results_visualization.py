"""
Analisador completo corrigido para extrair métricas dos locais corretos e criar análises completas.
"""

import re
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

warnings.filterwarnings('ignore')

# Configuração de estilo
plt.style.use('default')
sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.figsize': (16, 10),
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'font.family': 'sans-serif',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

class CompleteResultsAnalyzer:
    def __init__(self, results_dir='./results', output_dir='./paper_figures'):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.class_names = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']
        self.cnn_models = ['VGG19', 'Inception', 'ResNet', 'Xception']
        self.ml_classifiers = ['randomforest', 'xgboost', 'adaboost', 'extratrees']

        self.color_palette = {
            'CNN': '#1f77b4',
            'Feature_Extraction': '#ff7f0e',
            'VGG19': '#2ca02c',
            'Inception': '#d62728',
            'ResNet': '#9467bd',
            'Xception': '#8c564b',
            'RandomForest': '#17becf',
            'XGBoost': '#bcbd22',
            'AdaBoost': '#ff7f0e',
            'ExtraTrees': '#e377c2'
        }

        self.results_data = {}

    def _parse_results_file_robust(self, file_path):
        """Parser robusto para extrair métricas."""
        metrics = {}

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Padrões mais específicos para diferentes formatos
            patterns = {
                'accuracy': [
                    r'Average.*?Accuracy:\s*([0-9]*\.?[0-9]+)',
                    r'Accuracy:\s*([0-9]*\.?[0-9]+)',
                    r'accuracy\s+([0-9]*\.?[0-9]+)'
                ],
                'precision': [
                    r'Average.*?Precision:\s*([0-9]*\.?[0-9]+)',
                    r'Precision:\s*([0-9]*\.?[0-9]+)',
                ],
                'recall': [
                    r'Average.*?Recall:\s*([0-9]*\.?[0-9]+)',
                    r'Recall:\s*([0-9]*\.?[0-9]+)'
                ],
                'f1_score': [
                    r'Average.*?F1.*?([0-9]*\.?[0-9]+)',
                    r'F1 Score:\s*([0-9]*\.?[0-9]+)',
                    r'F1:\s*([0-9]*\.?[0-9]+)'
                ]
            }

            # Primeiro tenta extrair com padrões diretos
            for metric, pattern_list in patterns.items():
                for pattern in pattern_list:
                    match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
                    if match:
                        try:
                            value = float(match.group(1))
                            metrics[metric] = value
                            break
                        except (ValueError, IndexError):
                            continue

            # Se não conseguiu extrair tudo, tenta com classification report
            if len(metrics) < 4:
                # Procura por macro avg na tabela de classification report
                macro_pattern = r'macro avg\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)'
                macro_match = re.search(macro_pattern, content)
                if macro_match:
                    precision_val = float(macro_match.group(1))
                    recall_val = float(macro_match.group(2))
                    f1_val = float(macro_match.group(3))

                    if 'precision' not in metrics:
                        metrics['precision'] = precision_val
                    if 'recall' not in metrics:
                        metrics['recall'] = recall_val
                    if 'f1_score' not in metrics:
                        metrics['f1_score'] = f1_val

            return metrics

        except Exception as e:
            print(f"❌ Erro ao processar {file_path}: {e}")
            return {}

    def _extract_cnn_model_name(self, dirname):
        """Extrai nome do modelo CNN do diretório."""
        for model in self.cnn_models:
            if model.lower() in dirname.lower():
                return model
        return "Unknown"

    def _extract_fe_model_name(self, dirname):
        """Extrai nome do extrator de features do diretório."""
        for model in self.cnn_models:
            if model.lower() in dirname.lower():
                return model
        return "Unknown"

    def collect_all_results(self):
        """Coleta todos os resultados dos experimentos."""
        print("🔍 Coletando resultados dos experimentos...")

        # Coleta resultados CNN
        self._collect_cnn_results()

        # Coleta resultados Feature Extraction
        self._collect_feature_extraction_results()

        print(f"✅ Resultados coletados: {len(self.results_data)} experimentos")

        # Mostra resumo
        cnn_count = len([d for d in self.results_data.values() if d['type'] == 'CNN'])
        fe_count = len([d for d in self.results_data.values() if d['type'] == 'Feature_Extraction'])

        cnn_valid = len([d for d in self.results_data.values()
                        if d['type'] == 'CNN' and d['cv_metrics'].get('f1_score', 0) > 0])
        fe_valid = len([d for d in self.results_data.values()
                       if d['type'] == 'Feature_Extraction' and d['cv_metrics'].get('f1_score', 0) > 0])

        print(f"  Experimentos CNN: {cnn_count}")
        print(f"  Experimentos Feature Extraction: {fe_count}")
        print(f"  Experimentos CNN válidos: {cnn_valid}")
        print(f"  Experimentos FE válidos: {fe_valid}")

    def _collect_cnn_results(self):
        """Coleta resultados CNN."""
        cnn_dirs = list(self.results_dir.glob('cnn_classifier_*'))
        print(f"  Encontrados {len(cnn_dirs)} diretórios CNN")

        for cnn_dir in cnn_dirs:
            model_name = self._extract_cnn_model_name(cnn_dir.name)
            print(f"    Processando CNN: {cnn_dir.name} -> {model_name}")

            # Busca overall_results.txt primeiro (cross-validation)
            overall_file = cnn_dir / 'overall_results.txt'
            if overall_file.exists():
                print(f"      Encontrado: {overall_file}")
                cv_metrics = self._parse_results_file_robust(overall_file)

                if cv_metrics and cv_metrics.get('f1_score', 0) > 0:
                    key = f"CNN_{model_name}"
                    self.results_data[key] = {
                        'type': 'CNN',
                        'model': model_name,
                        'cv_metrics': cv_metrics,
                        'path': cnn_dir
                    }
                    print(f"      ✅ CNN {model_name} adicionado com métricas: {cv_metrics}")
                else:
                    print(f"      ⚠️ Métricas inválidas para CNN {model_name}")
            else:
                print(f"      ⚠️ Arquivo overall_results.txt não encontrado para {model_name}")

    def _collect_feature_extraction_results(self):
        """Coleta resultados Feature Extraction."""
        fe_dirs = list(self.results_dir.glob('feature_extraction_*'))
        print(f"  Encontrados {len(fe_dirs)} diretórios Feature Extraction")

        for fe_dir in fe_dirs:
            extractor_name = self._extract_fe_model_name(fe_dir.name)
            print(f"    Processando FE: {fe_dir.name} -> {extractor_name}")

            # Procura por cada classificador
            for classifier_name in self.ml_classifiers:
                classifier_dir = fe_dir / classifier_name

                if classifier_dir.exists():
                    print(f"      Verificando classificador: {classifier_name}")

                    # Busca overall_results.txt no diretório do classificador
                    overall_file = classifier_dir / 'overall_results.txt'
                    if overall_file.exists():
                        print(f"        Encontrado: {overall_file}")
                        cv_metrics = self._parse_results_file_robust(overall_file)

                        if cv_metrics and cv_metrics.get('f1_score', 0) > 0:
                            key = f"FE_{extractor_name}_{classifier_name.title()}"
                            self.results_data[key] = {
                                'type': 'Feature_Extraction',
                                'extractor': extractor_name,
                                'classifier': classifier_name.title(),
                                'cv_metrics': cv_metrics,
                                'path': classifier_dir
                            }
                            print(f"        ✅ FE {extractor_name}+{classifier_name.title()} adicionado: {cv_metrics}")
                        else:
                            print(f"        ⚠️ Métricas inválidas para {extractor_name}+{classifier_name}")
                    else:
                        print(f"        ⚠️ overall_results.txt não encontrado para {classifier_name}")

    def evaluate_final_models_on_test(self):
        """Avalia modelos finais no conjunto de teste usando features já extraídas."""
        print("🧪 Avaliando modelos finais no conjunto de teste...")

        # Para cada Feature Extraction, carrega features de teste e avalia modelos finais
        fe_dirs = list(self.results_dir.glob('feature_extraction_*'))

        for fe_dir in fe_dirs:
            extractor_name = self._extract_fe_model_name(fe_dir.name)

            # Carrega features de teste
            test_features_file = fe_dir / 'features' / 'test_features.npz'
            if not test_features_file.exists():
                print(f"  ⚠️ Features de teste não encontradas para {extractor_name}")
                continue

            try:
                test_data = np.load(test_features_file, allow_pickle=True)
                test_features = test_data['features']
                test_labels = test_data['labels']
                print(f"  ✅ Features de teste carregadas para {extractor_name}: {test_features.shape}")

                # Avalia cada classificador
                for classifier_name in self.ml_classifiers:
                    classifier_dir = fe_dir / classifier_name
                    final_model_file = classifier_dir / 'final_model' / 'final_ml_model.joblib'

                    if final_model_file.exists():
                        try:
                            # Carrega modelo final
                            model = joblib.load(final_model_file)

                            # Faz predições
                            y_pred = model.predict(test_features)

                            # Calcula métricas
                            accuracy = accuracy_score(test_labels, y_pred)
                            precision, recall, f1, _ = precision_recall_fscore_support(
                                test_labels, y_pred, average='macro', zero_division=0
                            )

                            # Armazena resultados
                            key = f"FE_{extractor_name}_{classifier_name.title()}"
                            if key in self.results_data:
                                self.results_data[key]['test_metrics'] = {
                                    'accuracy': accuracy,
                                    'precision': precision,
                                    'recall': recall,
                                    'f1_score': f1
                                }
                                print(f"    ✅ {classifier_name.title()}: Acc={accuracy:.3f}, F1={f1:.3f}")

                        except Exception as e:
                            print(f"    ❌ Erro ao avaliar {classifier_name}: {e}")

            except Exception as e:
                print(f"  ❌ Erro ao carregar features de {extractor_name}: {e}")

    def plot_cnn_vs_fe_by_network(self):
        """Gráfico comparativo CNN vs Feature Extraction por rede - O QUE VOCÊ PEDIU!"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('CNN End-to-End vs Feature Extraction + ML por Rede Neural',
                     fontsize=16, fontweight='bold', y=0.95)

        # Prepara dados organizados por rede
        network_data = {}

        for key, data in self.results_data.items():
            if data['type'] == 'CNN':
                network = data['model']
                if network not in network_data:
                    network_data[network] = {'CNN': None, 'FE': []}
                network_data[network]['CNN'] = data['cv_metrics']

            elif data['type'] == 'Feature_Extraction':
                network = data['extractor']
                if network not in network_data:
                    network_data[network] = {'CNN': None, 'FE': []}

                fe_result = {
                    'classifier': data['classifier'],
                    'metrics': data['cv_metrics']
                }
                network_data[network]['FE'].append(fe_result)

        # 1. Comparação F1-Score por rede
        ax1 = axes[0, 0]
        networks = list(network_data.keys())
        x = np.arange(len(networks))
        width = 0.35

        cnn_f1_scores = []
        fe_best_f1_scores = []
        fe_avg_f1_scores = []

        for network in networks:
            # CNN F1-Score
            cnn_f1 = network_data[network]['CNN']['f1_score'] if network_data[network]['CNN'] else 0
            cnn_f1_scores.append(cnn_f1)

            # Feature Extraction F1-Scores
            fe_f1_list = [fe['metrics']['f1_score'] for fe in network_data[network]['FE']
                         if fe['metrics'].get('f1_score', 0) > 0]

            if fe_f1_list:
                fe_best_f1_scores.append(max(fe_f1_list))
                fe_avg_f1_scores.append(np.mean(fe_f1_list))
            else:
                fe_best_f1_scores.append(0)
                fe_avg_f1_scores.append(0)

        bars1 = ax1.bar(x - width/2, cnn_f1_scores, width, label='CNN End-to-End',
                       color=self.color_palette['CNN'], alpha=0.8)
        bars2 = ax1.bar(x + width/2, fe_best_f1_scores, width, label='FE + ML (Melhor)',
                       color=self.color_palette['Feature_Extraction'], alpha=0.8)

        ax1.set_xlabel('Rede Neural')
        ax1.set_ylabel('F1-Score')
        ax1.set_title('F1-Score: CNN vs Feature Extraction (Melhor)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(networks)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Adiciona valores nas barras
        for bar, value in zip(bars1, cnn_f1_scores):
            if value > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        for bar, value in zip(bars2, fe_best_f1_scores):
            if value > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # 2. Detalhamento por classificador para cada rede
        ax2 = axes[0, 1]

        # Cria dados para heatmap: Rede vs Classificador
        classifiers_used = set()
        for network_info in network_data.values():
            for fe in network_info['FE']:
                classifiers_used.add(fe['classifier'])

        classifiers_used = sorted(list(classifiers_used))

        if classifiers_used:
            heatmap_data = np.zeros((len(networks), len(classifiers_used)))

            for i, network in enumerate(networks):
                for j, classifier in enumerate(classifiers_used):
                    # Encontra F1-score para esta combinação
                    for fe in network_data[network]['FE']:
                        if fe['classifier'] == classifier:
                            heatmap_data[i, j] = fe['metrics'].get('f1_score', 0)
                            break

            im = ax2.imshow(heatmap_data, cmap='viridis', aspect='auto')
            ax2.set_xticks(np.arange(len(classifiers_used)))
            ax2.set_yticks(np.arange(len(networks)))
            ax2.set_xticklabels(classifiers_used, rotation=45)
            ax2.set_yticklabels(networks)

            # Adiciona valores nas células
            for i in range(len(networks)):
                for j in range(len(classifiers_used)):
                    if heatmap_data[i, j] > 0:
                        text = ax2.text(j, i, f'{heatmap_data[i, j]:.3f}',
                                       ha="center", va="center", color="white",
                                       fontweight='bold', fontsize=8)

            ax2.set_title('F1-Score: Feature Extraction por Rede + Classificador')
            ax2.set_xlabel('Classificador ML')
            ax2.set_ylabel('Rede Neural (Extrator)')

            # Colorbar
            cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
            cbar.set_label('F1-Score', rotation=270, labelpad=15)

        # 3. Diferença relativa CNN vs FE
        ax3 = axes[1, 0]

        differences = []
        diff_labels = []
        colors = []

        for network in networks:
            cnn_f1 = network_data[network]['CNN']['f1_score'] if network_data[network]['CNN'] else 0
            fe_f1_list = [fe['metrics']['f1_score'] for fe in network_data[network]['FE']
                         if fe['metrics'].get('f1_score', 0) > 0]

            if fe_f1_list and cnn_f1 > 0:
                best_fe_f1 = max(fe_f1_list)
                diff = best_fe_f1 - cnn_f1  # Positivo = FE melhor, Negativo = CNN melhor
                differences.append(diff)
                diff_labels.append(network)
                colors.append('green' if diff > 0 else 'red')

        if differences:
            bars = ax3.barh(diff_labels, differences, color=colors, alpha=0.7)
            ax3.set_xlabel('Diferença F1-Score (FE - CNN)')
            ax3.set_title('Vantagem Feature Extraction vs CNN por Rede')
            ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            ax3.grid(True, alpha=0.3)

            # Adiciona valores
            for bar, diff in zip(bars, differences):
                ax3.text(diff + (0.005 if diff > 0 else -0.005), bar.get_y() + bar.get_height()/2,
                        f'{diff:+.3f}', ha='left' if diff > 0 else 'right', va='center',
                        fontsize=9, fontweight='bold')

        # 4. Ranking detalhado
        ax4 = axes[1, 1]

        # Coleta todos os resultados para ranking
        all_results = []

        for network in networks:
            # CNN
            if network_data[network]['CNN']:
                cnn_f1 = network_data[network]['CNN']['f1_score']
                all_results.append({
                    'name': f'CNN-{network}',
                    'f1_score': cnn_f1,
                    'type': 'CNN',
                    'network': network
                })

            # Feature Extraction
            for fe in network_data[network]['FE']:
                if fe['metrics'].get('f1_score', 0) > 0:
                    all_results.append({
                        'name': f'{network}+{fe["classifier"][:4]}',
                        'f1_score': fe['metrics']['f1_score'],
                        'type': 'FE',
                        'network': network
                    })

        # Ordena por F1-Score
        all_results.sort(key=lambda x: x['f1_score'], reverse=True)

        # Pega top 10
        top_results = all_results[:10]

        if top_results:
            names = [r['name'] for r in top_results]
            f1_scores = [r['f1_score'] for r in top_results]
            result_colors = [self.color_palette['CNN'] if r['type'] == 'CNN'
                           else self.color_palette['Feature_Extraction'] for r in top_results]

            bars = ax4.barh(range(len(names)), f1_scores, color=result_colors, alpha=0.8)
            ax4.set_yticks(range(len(names)))
            ax4.set_yticklabels(names, fontsize=9)
            ax4.set_xlabel('F1-Score')
            ax4.set_title('Top 10 Modelos por F1-Score')
            ax4.grid(True, alpha=0.3)

            # Adiciona valores
            for i, (bar, score) in enumerate(zip(bars, f1_scores)):
                ax4.text(score + 0.005, i, f'{score:.3f}', va='center', fontsize=8)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(self.output_dir / 'cnn_vs_fe_by_network.png')
        plt.close()

    def generate_comprehensive_report(self):
        """Gera relatório abrangente com todos os resultados."""
        report_path = self.output_dir / 'comprehensive_analysis_report.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RELATÓRIO ABRANGENTE - CLASSIFICAÇÃO DE CÂNCER DE PELE\n")
            f.write("CNN End-to-End vs Feature Extraction + ML\n")
            f.write("="*80 + "\n\n")

            # Estatísticas gerais
            f.write("1. ESTATÍSTICAS GERAIS\n")
            f.write("-"*40 + "\n")
            f.write(f"Total de experimentos: {len(self.results_data)}\n")

            cnn_count = len([d for d in self.results_data.values() if d['type'] == 'CNN'])
            fe_count = len([d for d in self.results_data.values() if d['type'] == 'Feature_Extraction'])

            f.write(f"Experimentos CNN: {cnn_count}\n")
            f.write(f"Experimentos Feature Extraction: {fe_count}\n\n")

            # Ranking completo
            f.write("2. RANKING COMPLETO\n")
            f.write("-"*40 + "\n")

            all_results = []
            for key, data in self.results_data.items():
                metrics = data['cv_metrics']
                f1 = metrics.get('f1_score', 0)
                acc = metrics.get('accuracy', 0)
                prec = metrics.get('precision', 0)
                rec = metrics.get('recall', 0)

                if data['type'] == 'CNN':
                    name = f"CNN-{data['model']}"
                else:
                    name = f"{data['extractor']}+{data['classifier']}"

                all_results.append((name, f1, acc, prec, rec, data['type']))

            # Ordena por F1-Score
            all_results.sort(key=lambda x: x[1], reverse=True)

            f.write("Ranking completo por F1-Score:\n")
            f.write("Pos. | Modelo                          | F1     | Acc    | Prec   | Rec    | Tipo\n")
            f.write("-"*85 + "\n")
            for i, (name, f1, acc, prec, rec, tipo) in enumerate(all_results, 1):
                f.write(f"{i:3d}. | {name:<30} | {f1:.4f} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {tipo}\n")

            f.write("\n")

            # Análise por tipo
            f.write("3. ANÁLISE POR TIPO\n")
            f.write("-"*40 + "\n")

            cnn_results = [r for r in all_results if r[5] == 'CNN']
            fe_results = [r for r in all_results if r[5] == 'Feature_Extraction']

            if cnn_results:
                cnn_f1_scores = [r[1] for r in cnn_results]
                f.write(f"CNN End-to-End ({len(cnn_results)} modelos):\n")
                f.write(f"  Média F1: {np.mean(cnn_f1_scores):.4f} ± {np.std(cnn_f1_scores):.4f}\n")
                f.write(f"  Melhor F1: {max(cnn_f1_scores):.4f}\n")
                f.write(f"  Pior F1: {min(cnn_f1_scores):.4f}\n\n")

            if fe_results:
                fe_f1_scores = [r[1] for r in fe_results]
                f.write(f"Feature Extraction + ML ({len(fe_results)} modelos):\n")
                f.write(f"  Média F1: {np.mean(fe_f1_scores):.4f} ± {np.std(fe_f1_scores):.4f}\n")
                f.write(f"  Melhor F1: {max(fe_f1_scores):.4f}\n")
                f.write(f"  Pior F1: {min(fe_f1_scores):.4f}\n\n")

            # Análise por rede neural
            f.write("4. ANÁLISE POR REDE NEURAL\n")
            f.write("-"*40 + "\n")

            network_analysis = {}
            for key, data in self.results_data.items():
                if data['type'] == 'CNN':
                    network = data['model']
                    if network not in network_analysis:
                        network_analysis[network] = {'CNN': [], 'FE': []}
                    network_analysis[network]['CNN'].append(data['cv_metrics']['f1_score'])

                elif data['type'] == 'Feature_Extraction':
                    network = data['extractor']
                    if network not in network_analysis:
                        network_analysis[network] = {'CNN': [], 'FE': []}
                    network_analysis[network]['FE'].append(data['cv_metrics']['f1_score'])

            for network, results in network_analysis.items():
                f.write(f"{network}:\n")
                if results['CNN']:
                    cnn_f1 = results['CNN'][0]  # Só há um CNN por rede
                    f.write(f"  CNN F1: {cnn_f1:.4f}\n")

                if results['FE']:
                    fe_f1_scores = results['FE']
                    f.write(f"  FE Média F1: {np.mean(fe_f1_scores):.4f} ± {np.std(fe_f1_scores):.4f}\n")
                    f.write(f"  FE Melhor F1: {max(fe_f1_scores):.4f}\n")
                    f.write(f"  FE Pior F1: {min(fe_f1_scores):.4f}\n")

                    # Vantagem FE vs CNN
                    if results['CNN']:
                        advantage = max(fe_f1_scores) - results['CNN'][0]
                        f.write(f"  Vantagem FE: {advantage:+.4f}\n")

                f.write("\n")

            # Conclusões
            f.write("5. CONCLUSÕES\n")
            f.write("-"*40 + "\n")

            if all_results:
                best_overall = all_results[0]
                f.write(f"• Melhor modelo geral: {best_overall[0]} (F1: {best_overall[1]:.4f})\n")

            # Melhor por categoria
            if cnn_results:
                best_cnn = max(cnn_results, key=lambda x: x[1])
                f.write(f"• Melhor CNN: {best_cnn[0]} (F1: {best_cnn[1]:.4f})\n")

            if fe_results:
                best_fe = max(fe_results, key=lambda x: x[1])
                f.write(f"• Melhor Feature Extraction: {best_fe[0]} (F1: {best_fe[1]:.4f})\n")

            # Melhor por rede neural
            for network, results in network_analysis.items():
                cnn_f1 = results['CNN'][0] if results['CNN'] else 0
                fe_best = max(results['FE']) if results['FE'] else 0

                if cnn_f1 > 0 or fe_best > 0:
                    if fe_best > cnn_f1:
                        f.write(f"• Melhor arquitetura {network}: Feature Extraction (F1: {fe_best:.4f})\n")
                    else:
                        f.write(f"• Melhor arquitetura {network}: CNN (F1 médio: {cnn_f1:.4f})\n")

            f.write("\n" + "="*80 + "\n")

        print(f"📋 Relatório abrangente salvo em: {report_path}")

    def plot_performance_overview(self):
        """Gráfico panorâmico da performance geral."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Visão Geral da Performance dos Modelos', fontsize=16, fontweight='bold', y=0.95)

        # Prepara dados
        cnn_data = {k: v for k, v in self.results_data.items() if v['type'] == 'CNN'}
        fe_data = {k: v for k, v in self.results_data.items() if v['type'] == 'Feature_Extraction'}

        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_names = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']

        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx // 2, idx % 2]

            # Coleta valores
            cnn_values = [v['cv_metrics'].get(metric, 0) for v in cnn_data.values()
                         if v['cv_metrics'].get(metric, 0) > 0]
            fe_values = [v['cv_metrics'].get(metric, 0) for v in fe_data.values()
                        if v['cv_metrics'].get(metric, 0) > 0]

            # Box plot
            data_to_plot = []
            labels = []
            colors = []

            if cnn_values:
                data_to_plot.append(cnn_values)
                labels.append('CNN')
                colors.append(self.color_palette['CNN'])

            if fe_values:
                data_to_plot.append(fe_values)
                labels.append('Feature Extraction')
                colors.append(self.color_palette['Feature_Extraction'])

            if data_to_plot:
                bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                               showmeans=True, meanline=True)

                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                # Adiciona estatísticas
                for i, values in enumerate(data_to_plot):
                    mean_val = np.mean(values)
                    ax.text(i+1, 0.95, f'μ={mean_val:.3f}',
                           ha='center', va='top', fontsize=10, fontweight='bold',
                           transform=ax.get_xaxis_transform())

            ax.set_title(f'{metric_name}', fontweight='bold', pad=10)
            ax.set_ylabel('Score')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(self.output_dir / 'performance_overview.png')
        plt.close()

    def run_complete_analysis(self):
        """Executa análise completa."""
        print("🚀 Iniciando análise completa...")

        # 1. Coleta resultados
        self.collect_all_results()

        # 2. Avalia modelos finais (se necessário)
        self.evaluate_final_models_on_test()

        # 3. Gera gráficos
        print("📊 Gerando gráficos...")
        self.plot_performance_overview()
        self.plot_cnn_vs_fe_by_network()  # O gráfico que você pediu!

        # 4. Gera relatório
        print("📋 Gerando relatório...")
        self.generate_comprehensive_report()

        print("✅ Análise completa finalizada!")
        print(f"📁 Resultados salvos em: {self.output_dir}")



# Função principal para executar
def main():
    """Função principal."""
    analyzer = CompleteResultsAnalyzer(
        results_dir='./results',
        output_dir='./paper_figures'
    )

    analyzer.run_complete_analysis()

    return analyzer

# Script executável
if __name__ == "__main__":
    main()