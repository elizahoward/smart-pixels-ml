#!/bin/bash

# Complete script to update all custom validation accuracy charts
# This ensures all Model1, Model2, and Model3 charts have consistent formatting

echo "🔧 Complete Custom Chart Formatting Update"
echo "==========================================="
echo "This script will:"
echo "1. Generate/regenerate custom validation accuracy charts for all models"
echo "2. Apply consistent formatting to match validation_accuracy_comparison.png"
echo "3. Create backups of original files"
echo ""

# Change to the script directory
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR"

echo "📍 Working directory: $(pwd)"
echo ""

# Step 1: Generate Model1 custom chart if needed
echo "📊 Step 1: Checking Model1 custom chart..."
MODEL1_DIR="Model1/quantized_model1_results_20250731_134522"
if [ -d "$MODEL1_DIR" ]; then
    echo "  Found Model1 results directory"
    if [ ! -f "$MODEL1_DIR/best_validation_accuracy_custom.png" ]; then
        echo "  Generating Model1 custom chart..."
        cd Model1
        python3 generate_validation_accuracy_chart.py quantized_model1_results_20250731_134522/
        cd ..
    else
        echo "  Model1 custom chart already exists"
    fi
else
    echo "  ⚠️  Model1 results directory not found: $MODEL1_DIR"
fi

echo ""

# Step 2: Apply consistent formatting to all custom charts
echo "📊 Step 2: Applying consistent formatting..."
python3 update_custom_chart_formatting.py

echo ""

# Step 3: Verify results
echo "📊 Step 3: Verification..."
echo "Checking for updated charts:"

UPDATED_COUNT=0

# Check Model1
if [ -f "Model1/quantized_model1_results_20250731_134522/best_validation_accuracy_custom.png" ]; then
    echo "  ✅ Model1: best_validation_accuracy_custom.png"
    ((UPDATED_COUNT++))
else
    echo "  ❌ Model1: best_validation_accuracy_custom.png NOT FOUND"
fi

# Check Model2
if [ -f "Model2/quantized_complicated_results_20250730_114020/best_validation_accuracy_custom.png" ]; then
    echo "  ✅ Model2: best_validation_accuracy_custom.png"
    ((UPDATED_COUNT++))
else
    echo "  ❌ Model2: best_validation_accuracy_custom.png NOT FOUND"
fi

# Check Model3
if [ -f "Model3/combined_results_20250730_021907/best_validation_accuracy_custom.png" ]; then
    echo "  ✅ Model3: best_validation_accuracy_custom.png"
    ((UPDATED_COUNT++))
else
    echo "  ❌ Model3: best_validation_accuracy_custom.png NOT FOUND"
fi

echo ""
echo "✅ Update complete! Successfully processed $UPDATED_COUNT model(s)."
echo ""
echo "📝 Applied formatting changes:"
echo "   - Figure size: Square (8.5 x 8.5) to match comparison charts"
echo "   - Title font size: 16 (consistent with validation_accuracy_comparison.png)"
echo "   - Axis label font size: 18"
echo "   - Tick label size: 14"
echo "   - Bar value font size: 18 (larger, readable labels)"
echo ""
echo "💾 Original files backed up as *_backup.png"
echo ""
echo "🎯 All best_validation_accuracy_custom.png files now have the same"
echo "   title, axis, and label sizes as validation_accuracy_comparison.png"