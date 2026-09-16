

### 1. Task Definition 
- Type: Multi-class classification (3 classes)
- Input: Transit-fit parameters and stellar properties for one Kepler Object of Interest (KOI), orbital period, transit depth/duration, planetary radius, stellar temperature. 
- Output koi_disposition one of CONFIRMED, CANIDATE, FALSE POSITIVE
- Unit of prediction: ONE KOI 
### 2. Offline metric
- Primary offline metric: Macro-average F1 across the 3 classes, avoids the model just learning to predict the majority class, treats getting CONFIRMED and CANDIDATE right as equally important to getting FALSE POSITIVE right.
- Secondary metrics: Per class recall, with special attention to recall on CONFIRMED.

### 3. Baseline and Target
- Majority class baseline - predicting the most common class only, to establish the accuracy floor. 
- Simple Heuristic Baseline: Logistic regression using only the 4 koi_fpflag_* Boolean columns. 
- Target:  Meaningfully beat the flag only baseline using the full feature set, targeting macro-F1 in the range of 0.8 - 0.90. 
### 4. Guardrail metrics
- Confusion between CONFIRMED and CANDIDATE specifically tracked separately from confusion with FALSE POSITIVE. 
- Calibration of the predicted probability since a probability ranked candidate list is how this actually be used to prioritize follow up observations. 
### 5. Constraints
- A tree based model is preferred so we can show feature importance. 
- No real latency constraints.
