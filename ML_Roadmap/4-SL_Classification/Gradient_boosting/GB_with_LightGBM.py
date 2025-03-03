import lightgbm as lgb

# 1️⃣ Create LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# 2️⃣ Define parameters
params = {
    'objective': 'binary',
    'metric': 'accuracy',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'max_depth': -1
}

# 3️⃣ Train LightGBM model
lgb_model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=100, verbose_eval=10)

# 4️⃣ Make predictions
y_pred_lgb = (lgb_model.predict(X_test) > 0.5).astype(int)

# 5️⃣ Evaluate the model
accuracy_lgb = accuracy_score(y_test, y_pred_lgb)
print(f"LightGBM Accuracy: {accuracy_lgb:.4f}")
