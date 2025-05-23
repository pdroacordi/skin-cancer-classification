"""
Script principal para análise completa dos resultados.
Versão simplificada que corrige os problemas de layout e usa features pré-extraídas.

USO:
    python run_analysis.py

SAÍDA:
    ./paper_figures/
    ├── performance_overview.png          # Figura 1 - Visão geral
    ├── detailed_comparison.png           # Figura 2 - Comparação detalhada
    ├── model_ranking.png                # Figura 3 - Ranking modelos
    ├── confusion_matrices_test.png      # Figura 4 - Matrizes confusão
    ├── statistical_analysis.png         # Figura 5 - Análise estatística
    ├── cnn_architecture_analysis.png    # Figura 6 - Arquiteturas CNN
    ├── ml_classifier_analysis.png       # Figura 7 - Classificadores ML
    ├── efficiency_analysis.png          # Figura 8 - Eficiência
    └── analysis_summary.txt             # Relatório textual
"""

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def check_requirements():
    """Verifica se os diretórios e arquivos necessários existem."""
    print("🔍 Verificando pré-requisitos...")

    issues = []

    # Verifica diretório de resultados
    results_dir = Path('./results')
    if not results_dir.exists():
        issues.append("Diretório './results' não encontrado")

    # Verifica experimentos CNN
    cnn_dirs = list(results_dir.glob('cnn_classifier_*')) if results_dir.exists() else []
    if not cnn_dirs:
        issues.append("Nenhum experimento CNN encontrado em ./results/")

    # Verifica experimentos Feature Extraction
    fe_dirs = list(results_dir.glob('feature_extraction_*')) if results_dir.exists() else []
    if not fe_dirs:
        issues.append("Nenhum experimento Feature Extraction encontrado em ./results/")

    # Verifica arquivo de teste
    test_file = Path('./res/test_files.txt')
    if not test_file.exists():
        issues.append("Arquivo './res/test_files.txt' não encontrado")

    # Verifica se há resultados overall
    overall_files = 0
    for cnn_dir in cnn_dirs:
        if (cnn_dir / 'overall_results.txt').exists():
            overall_files += 1

    for fe_dir in fe_dirs:
        for classifier in ['randomforest', 'xgboost', 'adaboost', 'extratrees']:
            if (fe_dir / classifier / 'overall_results.txt').exists():
                overall_files += 1

    if overall_files == 0:
        issues.append("Nenhum arquivo 'overall_results.txt' encontrado")

    # Verifica features pré-extraídas
    features_found = 0
    for fe_dir in fe_dirs:
        if (fe_dir / 'features' / 'test_features.npz').exists():
            features_found += 1

    if issues:
        print("❌ Problemas encontrados:")
        for issue in issues:
            print(f"   • {issue}")
        print("\n💡 Dicas:")
        print("   • Execute os experimentos primeiro (main.py)")
        print("   • Certifique-se de que a validação cruzada foi executada")
        print("   • Verifique se os modelos finais foram treinados")
        return False
    else:
        print("✅ Pré-requisitos OK:")
        print(f"   • {len(cnn_dirs)} experimentos CNN")
        print(f"   • {len(fe_dirs)} experimentos Feature Extraction")
        print(f"   • {overall_files} arquivos de resultados")
        print(f"   • {features_found} conjuntos de features pré-extraídas")
        return True

def main():
    """Função principal."""
    print("🚀 ANÁLISE DE RESULTADOS - CLASSIFICAÇÃO DE CÂNCER DE PELE")
    print("=" * 60)

    # Verifica pré-requisitos
    if not check_requirements():
        print("\n❌ Corrija os problemas antes de continuar.")
        sys.exit(1)

    # Importa e executa análise principal
    try:
        print("\n📊 Executando análise principal...")

        # Importa localmente para evitar problemas se módulos não existirem
        sys.path.append('.')
        from results_visualization import SkinCancerResultsAnalyzer
        from classifier_analyisis import ClassifierAnalyzer

        # Cria analisador principal
        analyzer = SkinCancerResultsAnalyzer(
            results_dir='./results',
            test_files_path='./res/test_files.txt',
            output_dir='./paper_figures'
        )

        # Coleta todos os resultados
        print("   Coletando resultados...")
        analyzer.collect_all_results()

        # Gera gráficos principais (sem mostrar)
        print("   Gerando gráficos principais...")
        analyzer.generate_all_plots()

        # Gera relatório
        print("   Gerando relatório...")
        analyzer.generate_summary_report()

        # Análise detalhada dos classificadores
        print("   Analisando classificadores...")
        classifier_analyzer = ClassifierAnalyzer(analyzer)
        classifier_analyzer.plot_cnn_architecture_comparison()
        classifier_analyzer.plot_ml_classifier_comparison()
        classifier_analyzer.plot_architecture_efficiency()

        # Resumo final
        print("\n" + "🎉" * 20)
        print("ANÁLISE CONCLUÍDA COM SUCESSO!")
        print("🎉" * 20)

        output_dir = Path('./paper_figures')
        print(f"\n📁 Gráficos salvos em: {output_dir.absolute()}")

        print("\n📊 Gráficos principais para o artigo:")
        graphs = [
            ("performance_overview.png", "Visão geral da performance"),
            ("detailed_comparison.png", "Comparação CNN vs Feature Extraction"),
            ("model_ranking.png", "Ranking dos melhores modelos"),
            ("confusion_matrices_test.png", "Matrizes de confusão (teste)"),
            ("statistical_analysis.png", "Análise estatística"),
            ("cnn_architecture_analysis.png", "Análise arquiteturas CNN"),
            ("ml_classifier_analysis.png", "Análise classificadores ML"),
            ("efficiency_analysis.png", "Análise de eficiência")
        ]

        for i, (filename, description) in enumerate(graphs, 1):
            filepath = output_dir / filename
            status = "✅" if filepath.exists() else "❌"
            print(f"   {status} Figura {i}: {filename} - {description}")

        report_file = output_dir / 'analysis_summary.txt'
        if report_file.exists():
            print(f"\n📋 Relatório textual: {report_file}")

        print(f"\n💡 Total de arquivos gerados: {len(list(output_dir.glob('*.png'))) + len(list(output_dir.glob('*.txt')))}")

    except ImportError as e:
        print(f"❌ Erro ao importar módulos: {e}")
        print("💡 Certifique-se de que todos os arquivos Python estão no mesmo diretório")
        sys.exit(1)

    except Exception as e:
        print(f"❌ Erro durante a análise: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()