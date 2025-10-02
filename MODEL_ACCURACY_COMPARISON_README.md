# Model Accuracy Comparison Visualization

## Tổng quan
Script `model_accuracy_comparison.py` được tạo để trực quan hóa và so sánh độ chính xác của các model machine learning trong hệ thống Cognitive Assessment System.

## Cách sử dụng

### Chạy script
```bash
python model_accuracy_comparison.py
```

### Kết quả
Script sẽ tạo ra:
1. **File biểu đồ**: `model_accuracy_comparison.png` - Biểu đồ so sánh độ chính xác
2. **Báo cáo console**: Hiển thị bảng tổng hợp độ chính xác của từng model

## Các Model trong hệ thống

### Classification Models (Đánh giá phân loại)
| Model | Độ chính xác | Mô tả |
|-------|-------------|--------|
| **RandomForest** | **65.4%** ⭐ | Model tốt nhất cho phân loại |
| XGBClassifier | 59.5% | XGBoost Classifier |
| StackingClassifier | 65.0% | Ensemble stacking |

### Regression Models (Đánh giá hồi quy - R² Score)
| Model | R² Score | Mô tả |
|-------|----------|--------|
| **StackingRegressor** | **-5.0%** | Tốt nhất (ít tệ nhất) |
| SVR | -520.0% | Rất tệ (worse than baseline) |
| XGBRegressor | -521.5% | Rất tệ |
| RandomForestRegressor | -523.5% | Tệ nhất |

**📝 Lưu ý về R² Score:**
- **R² > 0**: Model tốt hơn baseline (trung bình)
- **R² = 0**: Model bằng baseline
- **R² < 0**: Model tệ hơn baseline
- **R² âm cao**: Model dự đoán rất tệ

### Multi-Model Ensemble Results
| Model | Độ chính xác | Mô tả |
|-------|-------------|--------|
| RandomForest | 80.8% | Random Forest ensemble |
| GradientBoost | 81.7% | Gradient Boosting |
| CrossValidation | 75.0% | Cross-validation result |

## Cấu trúc biểu đồ

Biểu đồ được tạo bao gồm 4 phần:
1. **Classification Models Accuracy**: So sánh độ chính xác các model phân loại
2. **Regression Models Performance**: So sánh hiệu suất các model hồi quy
3. **Multi-Model Ensemble Results**: Kết quả ensemble models
4. **Combined Model Comparison**: So sánh tổng hợp tất cả models

## Tính năng

### ✨ Tự động tải dữ liệu
- Script tự động tìm và tải kết quả training từ:
  - `backend/results/training_results_comprehensive.json`
  - `backend/results/training_results_multi_20250830_211239.json`

### 📊 Trực quan hóa đẹp
- Sử dụng matplotlib và seaborn với style hiện đại
- Highlight model tốt nhất
- Hiển thị giá trị trên mỗi cột
- Grid lines và màu sắc phù hợp

### 📋 Báo cáo chi tiết
- Bảng tổng hợp độ chính xác
- Xếp hạng models theo hiệu suất
- Thông tin model tốt nhất

### 🔧 Linh hoạt
- Tự động fallback sang dữ liệu mẫu nếu không tìm thấy file kết quả
- Có thể tùy chỉnh đường dẫn file đầu vào
- Lưu biểu đồ với độ phân giải cao (300 DPI)

## Yêu cầu hệ thống

```bash
pip install matplotlib seaborn pandas numpy
```

## Output Files
- `model_accuracy_comparison.png`: Biểu đồ so sánh độ chính xác (487KB)

## Model Performance Summary

**🏆 Model tốt nhất tổng thể**: GradientBoost với 81.7% accuracy (ensemble)
**🥈 Classification**: RandomForest với 65.4% accuracy
**🥉 Regression**: StackingRegressor với -5.0% R² (ít tệ nhất trong nhóm)

**📈 Phân tích**:
- **Classification**: RandomForest đạt 65.4% - khá tốt
- **Regression**: TẤT CẢ models đều tệ hơn baseline (R² âm)
  - StackingRegressor: ít tệ nhất (-5%)
  - Các model khác: cực kỳ tệ (-520% đến -524%)
- **Ensemble**: GradientBoost đạt 81.7% - tốt nhất tổng thể

**⚠️ Vấn đề quan trọng**: Regression models hoạt động rất tệ, tệ hơn cả việc đoán giá trị trung bình. Cần cải thiện đáng kể!

## Lưu ý
- Đồ thị được tối ưu hóa cho màn hình có độ phân giải cao
- Có thể mở rộng để thêm nhiều model và metrics khác
- Script tương thích với dữ liệu training results hiện tại của hệ thống
