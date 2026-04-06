#!/bin/bash
# Master Pipeline Script for Clustering Statistical Analysis
# 
# This script runs the complete analysis pipeline:
# 1. Extract clustering metrics from embeddings
# 2. Compute statistical significance
# 3. Generate visualization plots
# 4. Generate LaTeX tables
#
# Usage:
#   bash run_analysis_pipeline.sh
#   bash run_analysis_pipeline.sh --base-dir auto_analysis --k 6

set -e  # Exit on error

# Default parameters
BASE_DIR="auto_analysis"
K=6
METHODS="bisecting_kmeans kmeans gmm"
OUTPUT_DIR="results"
FIGURES_DIR="figures"
TABLES_DIR="tables"
EXCLUDE=""
PRESERVE_FROM=""

# Parse command line arguments
while [ $# -gt 0 ]; do
    case $1 in
        --base-dir)
            BASE_DIR="$2"
            shift 2
            ;;
        --k)
            K="$2"
            shift 2
            ;;
        --exclude)
            EXCLUDE="$2"
            shift 2
            ;;
        --preserve-from)
            PRESERVE_FROM="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: bash run_analysis_pipeline.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --base-dir DIR      Base directory with experiments (default: auto_analysis)"
            echo "  --k NUM             Number of clusters (default: 6)"
            echo "  --exclude \"STRS\"    Exclude models from retraining (e.g. \"no_eka no_kan\")"
            echo "  --preserve-from FILE Existing raw metrics CSV to preserve excluded models from"
            echo "  --output-dir DIR    Output directory (default: results)"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "===================================================================================================="
echo "🚀 CLUSTERING STATISTICAL ANALYSIS PIPELINE"
echo "===================================================================================================="
echo "Base directory: $BASE_DIR"
echo "Number of clusters (k): $K"
echo "Exclude from retraining: $EXCLUDE"
echo "Preserve results from: $PRESERVE_FROM"
echo "Output directory: $OUTPUT_DIR"
echo "===================================================================================================="
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$FIGURES_DIR"
mkdir -p "$TABLES_DIR"

# Step 1: Run clustering analysis
echo "===================================================================================================="
echo "📊 STEP 1: Running Clustering Analysis"
echo "===================================================================================================="
EXCLUDE_CMD=""
if [ ! -z "$EXCLUDE" ]; then
    EXCLUDE_CMD="--model-exclude $EXCLUDE"
fi

PRESERVE_CMD=""
if [ ! -z "$PRESERVE_FROM" ]; then
    PRESERVE_CMD="--preserve-from $PRESERVE_FROM"
fi

python run_clustering_analysis.py \
    --base-dir "$BASE_DIR" \
    --k $K \
    --methods $METHODS \
    --output "$OUTPUT_DIR/raw_clustering_metrics.csv" \
    $EXCLUDE_CMD \
    $PRESERVE_CMD

if [ $? -ne 0 ]; then
    echo "❌ Step 1 failed!"
    exit 1
fi

echo ""

# Step 2: Compute statistical significance
echo "===================================================================================================="
echo "🧪 STEP 2: Computing Statistical Significance"
echo "===================================================================================================="
python compute_statistical_significance.py \
    --input "$OUTPUT_DIR/raw_clustering_metrics.csv" \
    --baseline baseline \
    --output "$OUTPUT_DIR/clustering_summary.csv" \
    --tests-output "$OUTPUT_DIR/statistical_tests.csv" \
    --latex-output "$TABLES_DIR/table_clustering.tex"

if [ $? -ne 0 ]; then
    echo "❌ Step 2 failed!"
    exit 1
fi

echo ""

# Step 3: Generate visualization plots
echo "===================================================================================================="
echo "📈 STEP 3: Generating Visualization Plots"
echo "===================================================================================================="
python generate_clustering_plots.py \
    --input "$OUTPUT_DIR/clustering_summary.csv" \
    --tests "$OUTPUT_DIR/statistical_tests.csv" \
    --output-dir "$FIGURES_DIR"

if [ $? -ne 0 ]; then
    echo "❌ Step 3 failed!"
    exit 1
fi

echo ""

# Step 4: Generate LaTeX tables
echo "===================================================================================================="
echo "📝 STEP 4: Generating LaTeX Tables"
echo "===================================================================================================="
python generate_latex_tables.py \
    --summary "$OUTPUT_DIR/clustering_summary.csv" \
    --tests "$OUTPUT_DIR/statistical_tests.csv" \
    --output-dir "$TABLES_DIR"

if [ $? -ne 0 ]; then
    echo "❌ Step 4 failed!"
    exit 1
fi

echo ""
echo "===================================================================================================="
echo "✅ PIPELINE COMPLETE!"
echo "===================================================================================================="
echo ""
echo "📁 Output Files:"
echo "   Raw metrics:          $OUTPUT_DIR/raw_clustering_metrics.csv"
echo "   Summary:              $OUTPUT_DIR/clustering_summary.csv"
echo "   Statistical tests:    $OUTPUT_DIR/statistical_tests.csv"
echo "   Figures:              $FIGURES_DIR/*.png"
echo "   LaTeX tables:         $TABLES_DIR/*.tex"
echo ""
echo "===================================================================================================="
