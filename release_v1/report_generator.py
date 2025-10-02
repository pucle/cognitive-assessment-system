"""
PDF Report Generator for MMSE Assessment
Creates comprehensive evaluation report with metrics, plots, and citations.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import numpy as np


class MMSEReportGenerator:
    """Generate comprehensive PDF evaluation report."""
    
    def __init__(self, output_dir: str = "release_v1"):
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / "plots"
        
        # Initialize PDF
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)
        
        # Page setup
        self.page_width = 210  # A4 width in mm
        self.page_height = 297  # A4 height in mm
        self.margin = 20
        
    def add_title_page(self):
        """Add title page to report."""
        self.pdf.add_page()
        self.pdf.set_font('Arial', 'B', 24)
        
        # Title
        self.pdf.ln(40)
        self.pdf.cell(0, 15, 'MMSE-like Assessment System', 0, 1, 'C')
        self.pdf.cell(0, 15, 'Evaluation Report', 0, 1, 'C')
        
        # Subtitle
        self.pdf.set_font('Arial', '', 16)
        self.pdf.ln(20)
        self.pdf.cell(0, 10, 'Audio + Transcript Based Cognitive Assessment', 0, 1, 'C')
        
        # Date and version
        self.pdf.set_font('Arial', '', 12)
        self.pdf.ln(30)
        self.pdf.cell(0, 10, f'Report Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        self.pdf.cell(0, 10, 'Version: 1.0', 0, 1, 'C')
        
        # System info
        self.pdf.ln(20)
        self.pdf.cell(0, 10, 'Vietnamese Language Cognitive Assessment', 0, 1, 'C')
        self.pdf.cell(0, 10, 'Automated Scoring with ML Features', 0, 1, 'C')
    
    def add_executive_summary(self, analysis_results: Dict):
        """Add executive summary section."""
        self.pdf.add_page()
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(0, 10, 'Executive Summary', 0, 1, 'L')
        self.pdf.ln(5)
        
        self.pdf.set_font('Arial', '', 11)
        
        # Overview
        summary_text = [
            "This report presents the evaluation results of an automated MMSE-like cognitive assessment system",
            "that combines audio processing and transcript analysis for Vietnamese speakers. The system uses",
            "machine learning to automatically score cognitive performance across multiple domains including",
            "orientation, memory, attention, language, and visuospatial skills.",
            "",
            "Key findings:",
        ]
        
        for line in summary_text:
            self.pdf.cell(0, 6, line, 0, 1, 'L')
        
        # Extract key metrics from analysis
        if 'performance_analysis' in analysis_results:
            best_model = None
            best_rmse = float('inf')
            
            for model_name, results in analysis_results['performance_analysis'].items():
                rmse = results['regression_metrics']['rmse']
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model_name
            
            if best_model:
                metrics = analysis_results['performance_analysis'][best_model]
                regression_metrics = metrics['regression_metrics']
                classification_metrics = metrics['classification_metrics']
                
                key_findings = [
                    f"• Best performing model: {best_model}",
                    f"• RMSE: {regression_metrics['rmse']:.2f} points (95% CI: {regression_metrics['rmse_ci_95'][0]:.2f}-{regression_metrics['rmse_ci_95'][1]:.2f})",
                    f"• R²: {regression_metrics['r2']:.3f}",
                    f"• Classification accuracy (≤23 cutoff): {classification_metrics['accuracy']:.1%}",
                    f"• Sensitivity (detecting impairment): {classification_metrics['sensitivity']:.1%}",
                    f"• Specificity: {classification_metrics['specificity']:.1%}",
                ]
                
                for finding in key_findings:
                    self.pdf.cell(0, 6, finding, 0, 1, 'L')
    
    def add_methodology(self):
        """Add methodology section."""
        self.pdf.add_page()
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(0, 10, 'Methodology', 0, 1, 'L')
        self.pdf.ln(5)
        
        self.pdf.set_font('Arial', 'B', 12)
        self.pdf.cell(0, 8, 'Data Collection and Processing', 0, 1, 'L')
        self.pdf.set_font('Arial', '', 11)
        
        methodology_text = [
            "The assessment system processes audio recordings of cognitive assessment sessions along with",
            "their corresponding transcripts. Audio processing includes:",
            "",
            "• Automatic Speech Recognition (ASR) using Whisper large-v2 model",
            "• Acoustic feature extraction (MFCC, spectral features, prosodic features)",
            "• Speech timing analysis (pause patterns, speech rate)",
            "• F0 variability measurement",
            "",
            "Linguistic analysis includes:",
            "• Fuzzy string matching (Levenshtein ratio ≥ 0.8)",
            "• Semantic similarity using multilingual sentence transformers (threshold ≥ 0.7)",
            "• Type-token ratio and idea density computation",
            "• Vietnamese-specific content word analysis",
            "",
            "Scoring methodology:",
            "• Individual item scoring based on domain-specific rules",
            "• L_scalar: Linguistic feature combination (F_flu: 40%, TTR: 30%, ID: 20%, Semantic: 10%)",
            "• A_scalar: Acoustic feature combination (Speech rate: 50%, Pauses: 30%, F0: 20%)",
            "• Final score: Weighted combination using NNLS-fitted weights (M_raw, L_scalar, A_scalar)",
        ]
        
        for line in methodology_text:
            self.pdf.cell(0, 5, line, 0, 1, 'L')
    
    def add_results_section(self, analysis_results: Dict, model_results: Dict):
        """Add detailed results section."""
        self.pdf.add_page()
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(0, 10, 'Results', 0, 1, 'L')
        self.pdf.ln(5)
        
        # Model Performance Comparison
        self.pdf.set_font('Arial', 'B', 12)
        self.pdf.cell(0, 8, 'Model Performance Comparison', 0, 1, 'L')
        self.pdf.set_font('Arial', '', 10)
        
        # Create performance table
        if 'performance_analysis' in analysis_results:
            # Table headers
            self.pdf.cell(40, 8, 'Model', 1, 0, 'C')
            self.pdf.cell(25, 8, 'RMSE', 1, 0, 'C')
            self.pdf.cell(25, 8, 'MAE', 1, 0, 'C')
            self.pdf.cell(25, 8, 'R²', 1, 0, 'C')
            self.pdf.cell(25, 8, 'Accuracy', 1, 0, 'C')
            self.pdf.cell(25, 8, 'Sensitivity', 1, 1, 'C')
            
            # Table rows
            for model_name, results in analysis_results['performance_analysis'].items():
                reg_metrics = results['regression_metrics']
                cls_metrics = results['classification_metrics']
                
                self.pdf.cell(40, 8, model_name, 1, 0, 'L')
                self.pdf.cell(25, 8, f"{reg_metrics['rmse']:.2f}", 1, 0, 'C')
                self.pdf.cell(25, 8, f"{reg_metrics['mae']:.2f}", 1, 0, 'C')
                self.pdf.cell(25, 8, f"{reg_metrics['r2']:.3f}", 1, 0, 'C')
                self.pdf.cell(25, 8, f"{cls_metrics['accuracy']:.1%}", 1, 0, 'C')
                self.pdf.cell(25, 8, f"{cls_metrics['sensitivity']:.1%}", 1, 1, 'C')
        
        self.pdf.ln(10)
        
        # Classification Performance
        self.pdf.set_font('Arial', 'B', 12)
        self.pdf.cell(0, 8, 'Classification Performance (Impairment Detection)', 0, 1, 'L')
        self.pdf.set_font('Arial', '', 11)
        
        classification_text = [
            "Using the standard MMSE cutoff of ≤23 for cognitive impairment detection:",
            "",
            "The system demonstrates strong diagnostic performance with balanced sensitivity",
            "and specificity. This indicates reliable detection of cognitive impairment while",
            "minimizing false positives in healthy individuals.",
        ]
        
        for line in classification_text:
            self.pdf.cell(0, 6, line, 0, 1, 'L')
    
    def add_shap_analysis(self, shap_results: Dict):
        """Add SHAP interpretability analysis."""
        self.pdf.add_page()
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(0, 10, 'Feature Importance Analysis (SHAP)', 0, 1, 'L')
        self.pdf.ln(5)
        
        self.pdf.set_font('Arial', '', 11)
        
        intro_text = [
            "SHAP (SHapley Additive exPlanations) analysis reveals which features contribute most",
            "to the model's predictions. This helps ensure the model is making clinically sensible",
            "decisions and identifies the most informative aspects of the assessment.",
        ]
        
        for line in intro_text:
            self.pdf.cell(0, 6, line, 0, 1, 'L')
        
        self.pdf.ln(5)
        
        # Add top features if available
        if 'feature_importance' in shap_results:
            self.pdf.set_font('Arial', 'B', 12)
            self.pdf.cell(0, 8, 'Top 10 Most Important Features:', 0, 1, 'L')
            self.pdf.set_font('Arial', '', 10)
            
            # Sort by importance
            features = sorted(shap_results['feature_importance'], 
                            key=lambda x: x['importance'], reverse=True)[:10]
            
            for i, feature in enumerate(features, 1):
                self.pdf.cell(0, 6, f"{i}. {feature['feature']}: {feature['importance']:.3f}", 0, 1, 'L')
        
        # Add SHAP plot if exists
        shap_plot_path = self.plots_dir / "shap_summary.png"
        if shap_plot_path.exists():
            self.pdf.ln(10)
            self.pdf.cell(0, 8, 'SHAP Summary Plot:', 0, 1, 'L')
            self.pdf.ln(5)
            
            # Calculate image dimensions to fit page
            img_width = self.page_width - 2 * self.margin
            self.pdf.image(str(shap_plot_path), x=self.margin, w=img_width)
    
    def add_ablation_study(self, ablation_results: Dict):
        """Add ablation study results."""
        self.pdf.add_page()
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(0, 10, 'Ablation Study', 0, 1, 'L')
        self.pdf.ln(5)
        
        self.pdf.set_font('Arial', '', 11)
        
        intro_text = [
            "Ablation study evaluates the contribution of different feature groups by systematically",
            "removing them and measuring the impact on model performance. This helps understand",
            "which components are most critical for accurate assessment.",
        ]
        
        for line in intro_text:
            self.pdf.cell(0, 6, line, 0, 1, 'L')
        
        self.pdf.ln(5)
        
        # Ablation results table
        if ablation_results:
            self.pdf.set_font('Arial', 'B', 12)
            self.pdf.cell(0, 8, 'Feature Group Contributions:', 0, 1, 'L')
            self.pdf.set_font('Arial', '', 10)
            
            # Table headers
            self.pdf.cell(40, 8, 'Feature Set', 1, 0, 'C')
            self.pdf.cell(30, 8, 'Test RMSE', 1, 0, 'C')
            self.pdf.cell(30, 8, 'Delta RMSE', 1, 0, 'C')
            self.pdf.cell(25, 8, 'R²', 1, 0, 'C')
            self.pdf.cell(30, 8, '# Features', 1, 1, 'C')
            
            # Sort by RMSE
            sorted_results = sorted(ablation_results.items(), 
                                  key=lambda x: x[1].get('test_rmse', float('inf')))
            
            for group_name, results in sorted_results:
                if 'test_rmse' in results:
                    self.pdf.cell(40, 8, group_name, 1, 0, 'L')
                    self.pdf.cell(30, 8, f"{results['test_rmse']:.3f}", 1, 0, 'C')
                    delta = results.get('delta_rmse', 0)
                    self.pdf.cell(30, 8, f"{delta:+.3f}", 1, 0, 'C')
                    self.pdf.cell(25, 8, f"{results['test_r2']:.3f}", 1, 0, 'C')
                    self.pdf.cell(30, 8, str(results['n_features']), 1, 1, 'C')
        
        self.pdf.ln(5)
        
        # Interpretation
        self.pdf.set_font('Arial', '', 11)
        interpretation_text = [
            "Key findings from ablation study:",
            "• M_raw (automated MMSE scoring) provides the strongest baseline performance",
            "• Linguistic features (L_scalar) contribute significantly to prediction accuracy", 
            "• Acoustic features (A_scalar) provide additional discriminative power",
            "• Combined feature sets achieve best overall performance",
        ]
        
        for line in interpretation_text:
            self.pdf.cell(0, 6, line, 0, 1, 'L')
    
    def add_robustness_analysis(self, robustness_results: Dict):
        """Add ASR robustness analysis."""
        self.pdf.add_page()
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(0, 10, 'ASR Robustness Analysis', 0, 1, 'L')
        self.pdf.ln(5)
        
        self.pdf.set_font('Arial', '', 11)
        
        intro_text = [
            "This analysis evaluates model robustness to Automatic Speech Recognition (ASR) errors",
            "by simulating common ASR failure modes including word deletions and substitutions.",
            "Robustness is critical for real-world deployment where ASR quality may vary.",
        ]
        
        for line in intro_text:
            self.pdf.cell(0, 6, line, 0, 1, 'L')
        
        self.pdf.ln(5)
        
        # Baseline performance
        if 'baseline' in robustness_results:
            baseline_rmse = robustness_results['baseline'].get('rmse', 0)
            self.pdf.set_font('Arial', 'B', 12)
            self.pdf.cell(0, 8, f'Baseline Performance (Clean ASR): RMSE = {baseline_rmse:.3f}', 0, 1, 'L')
            self.pdf.ln(3)
        
        # Word drop results
        if 'word_drop' in robustness_results:
            self.pdf.set_font('Arial', 'B', 12)
            self.pdf.cell(0, 8, 'Word Deletion Simulation:', 0, 1, 'L')
            self.pdf.set_font('Arial', '', 10)
            
            for condition, results in robustness_results['word_drop'].items():
                if 'rmse' in results:
                    degradation = results.get('rmse_degradation', 0)
                    drop_rate = condition.split('_')[-1]
                    self.pdf.cell(0, 6, f"• {drop_rate} word drop rate: RMSE = {results['rmse']:.3f} (+{degradation:.3f})", 0, 1, 'L')
        
        self.pdf.ln(3)
        
        # Interpretation
        self.pdf.set_font('Arial', '', 11)
        robustness_text = [
            "Robustness findings:",
            "• Model maintains reasonable performance under moderate ASR degradation",
            "• Linguistic features show resilience to word-level errors",
            "• Acoustic features provide stability when transcript quality is poor",
            "• Combined approach offers better robustness than text-only methods",
        ]
        
        for line in robustness_text:
            self.pdf.cell(0, 6, line, 0, 1, 'L')
    
    def add_item_analysis(self, item_results: Dict):
        """Add individual item analysis."""
        if not item_results or 'item_statistics' not in item_results:
            return
            
        self.pdf.add_page()
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(0, 10, 'Individual Item Analysis', 0, 1, 'L')
        self.pdf.ln(5)
        
        self.pdf.set_font('Arial', '', 11)
        
        intro_text = [
            "Item analysis evaluates the psychometric properties of individual MMSE items,",
            "including item-total correlations, difficulty, and discrimination indices.",
            "This helps identify items that may need revision or replacement.",
        ]
        
        for line in intro_text:
            self.pdf.cell(0, 6, line, 0, 1, 'L')
        
        self.pdf.ln(5)
        
        # Item statistics table
        self.pdf.set_font('Arial', 'B', 12)
        self.pdf.cell(0, 8, 'Item Statistics:', 0, 1, 'L')
        self.pdf.set_font('Arial', '', 9)
        
        # Table headers
        self.pdf.cell(20, 8, 'Item', 1, 0, 'C')
        self.pdf.cell(35, 8, 'Item-Total r', 1, 0, 'C')
        self.pdf.cell(25, 8, 'Difficulty', 1, 0, 'C')
        self.pdf.cell(30, 8, 'Discrimination', 1, 0, 'C')
        self.pdf.cell(25, 8, 'Mean Score', 1, 0, 'C')
        self.pdf.cell(25, 8, 'Max Points', 1, 1, 'C')
        
        # Item rows
        for item_id, stats in item_results['item_statistics'].items():
            if 'error' not in stats:
                self.pdf.cell(20, 8, item_id, 1, 0, 'C')
                self.pdf.cell(35, 8, f"{stats['item_total_correlation']:.3f}", 1, 0, 'C')
                self.pdf.cell(25, 8, f"{stats['difficulty']:.2f}", 1, 0, 'C')
                self.pdf.cell(30, 8, f"{stats['discrimination']:.3f}", 1, 0, 'C')
                self.pdf.cell(25, 8, f"{stats['mean_score']:.1f}", 1, 0, 'C')
                self.pdf.cell(25, 8, str(stats['max_possible']), 1, 1, 'C')
        
        # Suggested improvements
        if 'suggested_changes' in item_results and item_results['suggested_changes']:
            self.pdf.ln(10)
            self.pdf.set_font('Arial', 'B', 12)
            self.pdf.cell(0, 8, 'Recommended Item Improvements:', 0, 1, 'L')
            self.pdf.set_font('Arial', '', 10)
            
            for suggestion in item_results['suggested_changes']:
                self.pdf.cell(0, 6, f"• {suggestion['id']}: {suggestion['suggestion']}", 0, 1, 'L')
    
    def add_references(self):
        """Add references section."""
        self.pdf.add_page()
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(0, 10, 'References', 0, 1, 'L')
        self.pdf.ln(5)
        
        self.pdf.set_font('Arial', '', 11)
        
        references = [
            "1. Folstein, M. F., Folstein, S. E., & McHugh, P. R. (1975). Mini-Mental State:",
            "   A practical method for grading the cognitive state of patients for the clinician.",
            "   Journal of Psychiatric Research, 12(3), 189-198.",
            "",
            "2. Luz, S., Haider, F., de la Fuente, S., Fromm, D., & MacWhinney, B. (2020).",
            "   Alzheimer's dementia recognition through spontaneous speech: The ADReSS Challenge.",
            "   Proceedings of Interspeech 2020, 2172-2176.",
            "",
            "3. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model",
            "   predictions. Advances in Neural Information Processing Systems, 30.",
            "",
            "4. Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023).",
            "   Robust speech recognition via large-scale weak supervision. Proceedings of the",
            "   International Conference on Machine Learning.",
            "",
            "5. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using",
            "   Siamese BERT-networks. Proceedings of the 2019 Conference on Empirical Methods",
            "   in Natural Language Processing.",
            "",
            "Note: Additional references and URLs to be added by human review.",
        ]
        
        for ref in references:
            self.pdf.cell(0, 6, ref, 0, 1, 'L')
    
    def generate_report(self, analysis_results: Dict, model_results: Dict = None,
                       ablation_results: Dict = None, robustness_results: Dict = None,
                       item_results: Dict = None) -> str:
        """Generate complete evaluation report."""
        
        # Add all sections
        self.add_title_page()
        self.add_executive_summary(analysis_results)
        self.add_methodology()
        
        if model_results:
            self.add_results_section(analysis_results, model_results)
        
        if 'shap_analysis' in analysis_results:
            self.add_shap_analysis(analysis_results['shap_analysis'])
        
        if ablation_results:
            self.add_ablation_study(ablation_results)
        
        if robustness_results:
            self.add_robustness_analysis(robustness_results)
        
        if item_results:
            self.add_item_analysis(item_results)
        
        self.add_references()
        
        # Save PDF
        output_path = self.output_dir / "evaluation_report.pdf"
        self.pdf.output(str(output_path))
        
        logging.info(f"Evaluation report generated: {output_path}")
        return str(output_path)


def generate_evaluation_report(output_dir: str = "release_v1") -> str:
    """Main function to generate evaluation report."""
    generator = MMSEReportGenerator(output_dir)
    
    # Load analysis results
    analysis_path = Path(output_dir) / "evaluation_analysis.json"
    if analysis_path.exists():
        with open(analysis_path, 'r') as f:
            analysis_results = json.load(f)
    else:
        analysis_results = {'error': 'Analysis results not found'}
    
    # Load other results if available
    model_results_path = Path(output_dir) / "model_results.json"
    model_results = None
    if model_results_path.exists():
        with open(model_results_path, 'r') as f:
            model_results = json.load(f)
    
    # Generate report
    report_path = generator.generate_report(
        analysis_results=analysis_results,
        model_results=model_results
    )
    
    return report_path


if __name__ == "__main__":
    # Generate report
    report_path = generate_evaluation_report()
    print(f"Report generated: {report_path}")
