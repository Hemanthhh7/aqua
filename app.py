if st.button("Analyze Feasibility"):
    lat, lon = LOCATIONS[state]

    df_live = add_water_yield(fetch_weather(lat, lon))
    X_live = df_live[["temperature", "humidity", "dew_point", "pressure"]]

    # ---- MODEL PREDICTIONS ----
    pred_xgb = xgb_model.predict(X_live)[-1]

    seq = X_live.values[-24:].reshape(1, 24, 4)
    pred_lstm = lstm_model.predict(seq)[0][0]

    final_pred = (pred_xgb + pred_lstm) / 2
    status = feasibility(final_pred)

    # ---- NEW OUTPUTS ----
    conf = confidence_score(pred_xgb, pred_lstm)
    sensitivity_df = get_sensitivity(xgb_model)

    # ---- DISPLAY OUTPUTS ----
    st.success(f"State: {state}")

    # 1️⃣ Water Yield
    st.metric("Predicted Water Yield (L/m²/day)", round(final_pred, 3))

    # 2️⃣ Feasibility Class
    st.info(f"Feasibility Class: {status}")

    # 3️⃣ Confidence Score
    st.metric("Prediction Confidence (%)", round(conf, 1))

    # 4️⃣ Sensitivity Index
    st.subheader("Sensitivity Index (Parameter Impact)")
    st.table(sensitivity_df)

    st.caption(f"XGBoost MAE: {round(mae_xgb, 4)}")

    st.subheader("Hourly Water Yield Trend")
    st.line_chart(df_live["water_yield"])
