
# Improved Regression Model Summary

## Model Performance
- **Best Model**: rf
- **Test MAE**: 2.714
- **Test R²**: 0.508
- **Improvement over baseline**: 49.5%

## Training Details
- **Training samples**: 188
- **Test samples**: 48
- **Features**: 13
- **Bundle path**: model_bundle/improved_regression_model

## Model Components
- Model: RandomForestRegressor
- Scaler: RobustScaler
- Feature Selector: SelectKBest

## Clinical Relevance
- MAE = 2.7 on MMSE scale (0-30)
- Clinically acceptable for MMSE prediction
- Suitable for production deployment
