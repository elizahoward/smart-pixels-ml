#!/bin/bash

# Script to update custom chart formatting to match validation accuracy comparison charts
# This will modify best_validation_accuracy_custom.png files in Model1, Model2, and Model3

echo "🔧 Updating Custom Chart Formatting"
echo "=================================="
echo "This script will update best_validation_accuracy_custom.png files"
echo "to have the same title, axis, and label sizes as validation_accuracy_comparison.png"
echo ""

# Change to the script directory
cd "$(dirname "$0")"

# Run the Python script
python3 update_custom_chart_formatting.py

echo ""
echo "✅ Formatting update complete!"
echo ""
echo "📊 The following changes were made:"
echo "   - Figure size: (9, 11) → (8.5, 8.5) [square plot]"
echo "   - Title font: 28 → 16"
echo "   - Y-axis label font: 36 → 18"
echo "   - X-axis tick font: 30 → 14"
echo "   - Y-axis tick font: 20 → 14"
echo "   - Bar value font: 24 → 12"
echo "   - Added X-axis label"
echo ""
echo "💾 Original files backed up as *_backup.png"