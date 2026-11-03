# Viva and Interview Questions: Restaurant Rating Prediction

Answers refer to the actual decisions and results of this project.

## A. Problem and data

**1. What problem did you solve and why is it regression?**
Predict a restaurant's aggregate rating (a continuous value, 0-5) from its other features. A continuous numeric target means regression, not classification.

**2. Describe the dataset.**
9,551 restaurants, 21 columns, 15 countries (about 90% India). Columns cover location (country, city, lat/long), cuisines, cost and currency, service flags, price range, votes, and the rating plus two labels derived from it. No duplicates; only 9 missing `Cuisines` values.

**3. What is target leakage? Where did it appear here?**
Leakage is when a feature contains information that would not be available at prediction time or that is derived from the target. `Rating color` and `Rating text` are bins of the rating (e.g. "Excellent" = 4.5-4.9). Keeping them would give near-perfect but meaningless scores, so I dropped both.

**4. Why did you exclude restaurants with rating 0?**
2,148 rows have 0, labelled "Not rated", and all have ≤3 votes. Zero means "no rating exists", not "very bad". Including them makes the model learn "few votes → 0", which inflated R² to 0.96 without any real improvement. I measured this explicitly and excluded them from the main model.

**5. How did you handle missing values and duplicates?**
Nine missing `Cuisines` were filled with "Unknown" (tiny share; keeps rows). There were no duplicates; I still call `drop_duplicates()` as a safeguard.

**6. Why did you drop columns like name, address and locality?**
They are identifiers or free text with very high cardinality (7,446 names, ~8,900 addresses, 1,208 localities); one-hot encoding them would explode dimensionality and invite overfitting. City plus latitude/longitude already capture location.

## B. Preprocessing and features

**7. Why is `Average Cost for two` problematic, and how did you fix it?**
It is in 12 currencies, so values are not comparable (e.g. Rupees vs Rupiah). I divide by the median cost of that currency (learned on training data only) and take the log, giving a unit-free "expensive relative to local market" feature. Zero-cost rows and unseen currencies get a neutral value.

**8. Why log-transform `Votes`?**
It ranges from 0 to ~10,900 and is heavily right-skewed. `log1p` compresses the tail (and handles 0 votes), which helps linear models and makes the relationship easier to see.

**9. How did you encode `Cuisines`?**
It is multi-valued, so ordinary one-hot encoding is wrong. I made `num_cuisines` plus 0/1 flags for the 15 most common cuisines, learned from training data.

**10. How did you encode `City` and `Country Code`?**
One-hot encoding with `min_frequency=20` (rare categories grouped) and `handle_unknown="infrequent_if_exist"` so unseen cities at prediction time don't crash the model.

**11. Why scale features? Do trees need it?**
Linear Regression benefits from scaled inputs; trees and forests are invariant to scaling. Applying `StandardScaler` uniformly in one pipeline is harmless to trees and keeps the code simple.

**12. How did you prevent data leakage in preprocessing?**
Split first, then fit everything inside a scikit-learn `Pipeline`. Encoders, scalers, top-cuisine list and median costs are learned only on training rows, and in cross-validation only on each training fold.

## C. Models and evaluation

**13. Why those three models?**
Linear Regression is a simple interpretable benchmark; Decision Tree captures non-linear rules and shows overfitting clearly; Random Forest averages many trees to reduce variance. I also added a mean-predicting baseline to show what "no skill" looks like.

**14. Explain MAE, MSE, RMSE and R².**
MAE: average absolute error (rating points). MSE: average squared error, penalises big misses. RMSE: √MSE, in rating units. R²: fraction of variance explained, 1 is perfect, 0 equals predicting the mean, negative is worse than the mean.

**15. When would you prefer RMSE over MAE?**
When large errors are especially costly, since squaring punishes them. MAE is more robust to outliers and easier to explain.

**16. What were your results?**
Tuned Random Forest on the test set: MAE 0.239, MSE 0.105, RMSE 0.323, R² 0.662. Linear Regression: R² 0.608. Tuned Decision Tree: R² 0.574. Mean baseline: R² 0.

**17. Why did the untuned Decision Tree do so badly?**
It grew until it memorised the training set (train R² 1.00) and generalised poorly (test R² 0.26): high variance, i.e. overfitting.

**18. Is your Random Forest overfitting?**
There is a gap (train R² 0.81 vs test 0.66), normal for forests. But test error is the lowest of all models, cross-validated RMSE (0.342) agrees with the test RMSE, and the learning curve shows how the gap behaves with more data. So it generalises best of the models tried, though not perfectly.

**19. How do you tell overfitting from underfitting?**
Overfitting: very low training error but much higher validation/test error. Underfitting: both errors are high. The tuned Decision Tree is closer to underfitting (train and test R² both ≈ 0.58).

**20. How did you choose the final model without cheating?**
By lowest 5-fold cross-validation RMSE on the training set. The test set was used once, for the final report.

## D. Tuning and improvement

**21. What did you tune and how?**
Decision Tree: grid search over `max_depth`, `min_samples_leaf`. Random Forest: randomized search (12 combinations) over `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`, all with 5-fold CV on the training data. Best forest: 300 trees, depth 20, min leaf 3, `max_features=0.3`.

**22. How much did tuning help?**
Modestly for the forest (R² 0.649 → 0.662; RMSE 0.329 → 0.323). Its biggest effect was fixing the Decision Tree's overfitting. I report this honestly rather than overstating it.

**23. Grid search vs randomized search?**
Grid tries every combination (thorough but exponential); randomized samples a fixed number of combinations (cheaper, often nearly as good). I used grid for the small tree space and randomized for the larger forest space.

**24. What would you try next to improve performance?**
Gradient boosting (e.g. `HistGradientBoostingRegressor`), richer cuisine/location features, per-country models, and review-text features if available.

## E. Interpretation

**25. Which features matter most?**
`Votes` by a large margin (shuffling it raises RMSE by 0.195), then Cuisines (0.039), City (0.026), latitude/longitude (~0.018 each). Service flags barely matter.

**26. Permutation importance vs built-in forest importance?**
Built-in (impurity) importance is computed during training and is biased toward continuous/high-cardinality features. Permutation importance measures the actual drop in test performance when a column is shuffled, so it is more reliable. I report both.

**27. Without `Votes` how good is the model?**
R² ≈ 0.54 (vs 0.66), so other features carry real signal, though Votes is the strongest.

**28. Is using `Votes` a problem?**
It is a legitimate input here (the task is to predict from other available features), but votes accumulate along with ratings, so it makes the task easier than rating a brand-new restaurant. I mention this as a limitation and quantify the no-Votes case.

**29. What do the residual plots tell you?**
Residuals are centred at 0 (mean 0.000, std 0.323) and roughly symmetric, with no strong pattern vs the prediction. The model tends to under-predict very high ratings (it regresses to the mean).

## F. Practical and conceptual

**30. Why save the whole pipeline instead of just the model?**
So prediction needs only raw columns; preprocessing can't drift out of sync with the model.

**31. What are the main limitations?**
~90% India data, popularity-linked `Votes`, no review text, unrated restaurants excluded, subjective target (R² ceiling well below 1).

**32. Explain bias-variance with your models.**
Linear Regression: higher bias, low variance (train ≈ test). Untuned tree: low bias, very high variance. Random Forest: averaging many trees cuts variance while keeping low bias, which is why it wins here.

**33. Why `random_state=42`?**
Reproducibility: the same split, same forest, same numbers every run.

**34. Why a Pipeline instead of manual steps?**
It guarantees identical preprocessing in training, cross-validation and prediction and prevents leakage by construction.

**35. If the interviewer says "R² of 0.66 is low", what do you say?**
Ratings are subjective and the data has no review text, only metadata. The model beats the mean baseline substantially (RMSE 0.323 vs 0.556, MAE about a quarter of a rating point), generalises consistently under cross-validation, and I verified the number isn't inflated by leakage or unrated rows. Higher scores I could have reported (0.96) were artefacts I deliberately avoided.
