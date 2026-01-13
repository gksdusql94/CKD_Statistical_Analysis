###############################################################
# =============================================================
# STEP 1) UNDERSTANDING THE DATASET
# =============================================================
###############################################################
library(dplyr)
library(readr)
library(tidyr)
library(ggplot2)
library(reshape2)
library(corrplot)
library(moments)
library(VIM)
library(fastDummies)
library(car)
library(MASS)

# -------------------------------------------------------------
# 1-1) Load Dataset
# -------------------------------------------------------------

df <- read_csv("/Users/han-yeonbi/Downloads/AMS572 Group Project/Data/chronic_kidney_disease.csv")
cat("First few rows of the dataset:\n")
print(sample_n(df, 5))

# -------------------------------------------------------------
# 1-2) Target Variable Distribution
# -------------------------------------------------------------
# Target distribution (%)
target_distribution <- prop.table(table(df$target)) * 100
cat("Target distribution (normalized):\n")
print(target_distribution)

# Count distribution
target_his <- table(df$target)
cat("Target distribution (count):\n")
print(target_his)

# Histogram
barplot(
  target_his,
  main = "Histogram of Target",
  xlab = "Target",
  ylab = "Count",
  col = "blue"
)

# -------------------------------------------------------------
# 1-3) Missing Values Overview
# -------------------------------------------------------------

# Data structure
summary(df)
dim(df)

# Identify categorical columns
categorical_cols <- names(df)[sapply(df, function(x) is.character(x) || is.factor(x))]
cat("Categorical columns:\n")
print(categorical_cols)

# Check unique values in categorical variables
for (col in categorical_cols) {
  cat("Variable:", col, "\n")
  print(unique(df[[col]]))
  cat("\n")
}

# Replace "?" with NA
df_mod <- df
df_mod[df_mod == "?" | df_mod == "\t?"] <- NA

# Show modified dataset
head(df_mod)

# Missing summary table
Total_Count <- nrow(df_mod)
Missing_Count <- colSums(is.na(df_mod))
Unique_Count <- sapply(df_mod, function(x) length(unique(x)))
Missing_Percentage <- (Missing_Count / Total_Count) * 100

df_summary <- data.frame(
  Total_Count,
  Missing_Count,
  Unique_Count,
  Missing_Percentage
)

df_summary

# Missing value barplot
barplot(
  Missing_Count,
  main = "Missing Value Count per Variable",
  xlab = "Variables",
  ylab = "Missing Count",
  names.arg = names(Missing_Count),
  col = "blue",
  las = 2
)

# Value counts for categorical variables
for (col in names(df_mod)[sapply(df_mod, is.character)]) {
  cat("Unique values & value counts for", col, ":\n")
  print(table(df_mod[[col]], useNA = "ifany"))
  cat("----------------------------------\n")
}

# -------------------------------------------------------------
# 1-4) Missing Values Overview
# -------------------------------------------------------------

df2 <- df_mod  # cleaned dataset

df2_numeric <- df2[sapply(df2, is.numeric)]
corr_numeric <- cor(df2_numeric, use = "complete.obs")

# Heatmap
corrplot(
  corr_numeric,
  method = "color",
  type = "full",
  addCoef.col = "black",
  number.cex = 0.7,
  tl.col = "black",
  tl.srt = 45,
  col = colorRampPalette(c("red", "white", "blue"))(200),
  title = "Heatmap of df2 (Numeric Only)",
  mar = c(0,0,2,0)
)

# High correlation pairs (threshold = 0.80)
corr_threshold <- 0.80
high_corr <- which(abs(corr_numeric) > corr_threshold & lower.tri(corr_numeric), arr.ind = TRUE)

high_corr_pairs <- data.frame(
  Var1 = rownames(corr_numeric)[high_corr[, 1]],
  Var2 = colnames(corr_numeric)[high_corr[, 2]],
  Correlation = corr_numeric[high_corr]
)

high_corr_pairs

# -------------------------------------------------------------
# 1-5) Missing Data Mechanisms (MCAR / MAR / MNAR Structure)
# -------------------------------------------------------------
df_ckd <- df2[df2$target == "ckd", ]
df_notckd <- df2[df2$target == "notckd", ]

missing_rate <- function(df) colMeans(is.na(df)) * 100

miss_ckd <- missing_rate(df_ckd)
miss_notckd <- missing_rate(df_notckd)

deltadiff <- miss_ckd - miss_notckd

missing_compare <- data.frame(
  Missing_in_CKD = round(miss_ckd, 2),
  Missing_in_notCKD = round(miss_notckd, 2),
  Difference = round(deltadiff, 2)
)

colnames(missing_compare) <- c(
  "Missing_in_CKD (%)",
  "Missing_in_notCKD (%)",
  "Δ Difference (%)"
)

missing_compare

###############################################################
# =============================================================
# STEP 2) HYPOTHESIS TESTING (H1: BP ~ APPETITE)
# =============================================================
###############################################################

# -------------------------------------------------------------
# 2-0) Missing Value Handling Strategy
# -------------------------------------------------------------
missvars_over_30 <- names(Missing_Percentage[Missing_Percentage > 30])
missvars_20_30 <- names(Missing_Percentage[
  Missing_Percentage >= 20 & Missing_Percentage < 30
])
missvars_10_20 <- names(Missing_Percentage[
  Missing_Percentage >= 10 & Missing_Percentage < 20
])
missvars_less_5 <- names(Missing_Percentage[Missing_Percentage < 5])

missing_groups_table <- data.frame(
  Variables = c(
    paste(missvars_less_5, collapse = ", "),
    paste(missvars_10_20, collapse = ", "),
    paste(missvars_20_30, collapse = ", "),
    paste(missvars_over_30, collapse = ", ")
  ),
  'Missing Rate' = c(
    "< 5%",
    "10–20%",
    "20–30%",
    "over 30%"
  ),
  'Handling Strategy' = c(
    "Drop missing rows",
    "KNN imputation",
    "KNN imputation (with caution)",
    "Excluded"
  )
)


# -------------------------------------------------------------
# 2-1) Hypotheses (H1: BP differs by Appetite)
# -------------------------------------------------------------
# H0: mean BP_good = mean BP_poor
# H1: mean BP_good != mean BP_poor

alpha <- 0.05

bp_good <- na.omit(df2$bp[df2$appet == "good"])
bp_poor <- na.omit(df2$bp[df2$appet == "poor"])

cat("Number of 'good appetite' patients:", length(bp_good), "\n")
cat("Number of 'poor appetite' patients:", length(bp_poor), "\n\n")


# -------------------------------------------------------------
# 2-2) F-test for Equality of Variances
# -------------------------------------------------------------
cat("--- F-test for Equality of Variances ---\n")

f_test_result <- var.test(bp_good, bp_poor)

cat("F-statistic:", round(f_test_result$statistic, 4), "\n")
cat("P-value:", formatC(f_test_result$p.value, format = "e", digits = 3), "\n")

equal_var <- f_test_result$p.value > alpha
cat("Variance equality assumed:", equal_var, "\n\n")


# -------------------------------------------------------------
# 2-3) Two-sample t-test (Welch or Equal variance)
# -------------------------------------------------------------
t_test_result <- t.test(bp_good, bp_poor, var.equal = equal_var)

t_stat <- round(t_test_result$statistic, 4)
p_val <- round(t_test_result$p.value, 4)

cat("H1 T-statistic:", t_stat, "\n")
cat("H1 P-value:", p_val, "\n\n")

if (t_test_result$p.value < alpha) {
  print("Result: Reject H0 — There is a significant difference in mean BP between appetite groups.")
} else {
  print("Result: Fail to reject H0 — No significant difference in mean BP between appetite groups.")
}


# -------------------------------------------------------------
# 2-4) Group Means
# -------------------------------------------------------------
cat("\nMean BP (good appetite):", round(mean(bp_good), 2), "\n")
cat("Mean BP (poor appetite):", round(mean(bp_poor), 2), "\n")


# -------------------------------------------------------------
# 2-5) Final Written Conclusion (H1)
# -------------------------------------------------------------
cat(
  "\nH1 Conclusion (bp vs. appetite):\n",
  "The t-test indicates a statistically significant association between appetite status and blood pressure.\n",
  "* T-statistic: ", t_stat, "\n",
  "* P-value: ", p_val, "\n",
  sep = ""
)


###############################################################
# H2) Multivariate Logistic Regression for CKD Prediction (R)
# Fully Reproducing the Python Pipeline
#
# 1) Variable selection
# 2) Dummy encoding
# 3) KNN imputation
# 4) VIF screening
# 5) Stepwise CV selection
# 6) Final logistic regression (L2-regularized)
# 7) Evaluation + OR table + ROC curve
###############################################################

###############################################################
# H2: Multivariate Logistic Regression (Python pipeline reproduced in R)
# Steps:
# 1) Select features
# 2) Dummy encoding
# 3) KNN imputation
# 4) VIF screening
# 5) Stepwise backward AIC selection
# 6) Fit initial 4-variable logistic regression
# 7) p-value diagnostics for each variable
# 8) Detect separation → remove 'al'
# 9) Fit final 3-variable logistic regression
# 10) OR table (odds ratios + CI)
# 11) Confusion Matrix (train/test)
# 12) ROC & AUC
# 13) Final summary text
###############################################################

library(dplyr)
library(fastDummies)
library(VIM)
library(car)
library(caret)
library(pROC)

library(dplyr)
library(fastDummies)
library(VIM)
library(car)
library(caret)
library(pROC)

###############################################################
# =============================================================
# 3-1) Feature Selection (same as Python)
# =============================================================
###############################################################

selected_features <- c(
  "age", "bp", "bgr", "hemo",      # Blood / Metabolism
  "al", "sg", "bu", "sc",          # Kidney function
  "appet", "htn", "dm", "ane",     # Clinical status
  "target"
)

df_h2 <- df2[, selected_features]

# Encode target: ckd=1, notckd=0
df_h2$target <- ifelse(df_h2$target == "ckd", 1,
                       ifelse(df_h2$target == "notckd", 0, NA))
df_h2 <- df_h2[!is.na(df_h2$target), ]
###############################################################
# =============================================================
# 3-2) Dummy Encoding (drop-first)
# =============================================================
###############################################################

df_h2$appet <- factor(df_h2$appet, levels=c("good","poor"))
df_h2$htn   <- factor(df_h2$htn,   levels=c("no","yes"))
df_h2$dm    <- factor(df_h2$dm,    levels=c("no","yes"))
df_h2$ane   <- factor(df_h2$ane,   levels=c("no","yes"))

df_h2_dummy <- fastDummies::dummy_cols(
  df_h2,
  select_columns = c("appet","htn","dm","ane"),
  remove_first_dummy = TRUE,
  remove_selected_columns = TRUE
)
###############################################################
# =============================================================
# 3-3) KNN Imputation
# =============================================================
###############################################################

knn_input <- df_h2_dummy

set.seed(572)
df_h2_knn <- VIM::kNN(
  knn_input,
  k = 5,
  imp_var = FALSE
)

# Remove *_NA dummy (all zeros)
df_h2_knn <- df_h2_knn[, !grepl("_NA$", names(df_h2_knn))]

# Remove constant columns
df_h2_knn <- df_h2_knn[, sapply(df_h2_knn, function(x) length(unique(x)) > 1)]


###############################################################
# =============================================================
# 3-4) VIF Screening (same logic as Python)
# =============================================================
###############################################################

lm_for_vif <- lm(target ~ ., data=df_h2_knn)
vif_values <- car::vif(lm_for_vif)

vif_table <- data.frame(
  Variable = names(vif_values),
  VIF = as.numeric(vif_values)
)

vif_table <- vif_table[order(-vif_table$VIF), ]
cat("\n--- VIF Table ---\n")
print(vif_table)

high_vif_vars <- vif_table$Variable[vif_table$VIF > 10]
cat("\nRemoved (VIF > 10):\n")
print(high_vif_vars)

df_reduced <- df_h2_knn[, !(names(df_h2_knn) %in% high_vif_vars)]

X_reduced <- subset(df_reduced, select = -target)
y_reduced <- df_reduced$target


###############################################################
# =============================================================
# 3-5) Stepwise Feature Selection (Backward AIC)
# =============================================================
###############################################################

ctrl <- rfeControl(
  functions = lrFuncs,
  method = "cv",
  number = 5,
  verbose = TRUE,
  returnResamp = "final",
  allowParallel = FALSE
)

set.seed(572)

full_model <- glm(target ~ ., data=df_reduced, family=binomial)

step_model <- step(
  full_model,
  direction = "backward",
  trace = TRUE
)

cat("\n--- Final Selected Variables (Backward AIC) ---\n")
final_vars <- names(coef(step_model))[-1]
print(final_vars)

# --- AIC Profile Visualization ---

aic_values <- step_model$anova$AIC   # Stepwise AIC path
steps <- seq_along(aic_values)

plot(
  steps, aic_values, type = "b",
  main = "AIC Profile Across Candidate Models",
  xlab = "Model Step",
  ylab = "AIC",
  pch = 21, bg = "white"
)

# lowest AIC 
min_idx <- which.min(aic_values)

points(
  min_idx, aic_values[min_idx],
  col = "red", pch = 19, cex = 1.5
)

abline(v = min_idx, col = "red", lty = 2)

text(
  min_idx, aic_values[min_idx],
  labels = "Lowest AIC", pos = 4, col = "red"
)


###############################################################
# =============================================================
# 3-6) p-value Diagnostics for Selected 4 Variables
# =============================================================
###############################################################

# Model without bgr
glm_nobgr <- glm(target ~ hemo + al + sg, data=df_reduced, family=binomial)
summary(glm_nobgr)

# Model without hemo
glm_nohemo <- glm(target ~ bgr + al + sg, data=df_reduced, family=binomial)
summary(glm_nohemo)

# Model without al
glm_noal <- glm(target ~ bgr + hemo + sg, data=df_reduced, family=binomial)
summary(glm_noal)

# Model without sg
glm_nosg <- glm(target ~ bgr + hemo + al, data=df_reduced, family=binomial)
summary(glm_nosg)

# Although the stepwise AIC procedure selected four predictors (bgr, hemo, al, sg),
# the p-value diagnostics revealed that the variable 'al' produced extremely large
# standard errors (>2000), z-values near zero, and p-values around 0.99.
# This indicates that the model is unable to estimate the effect of 'al',
# and the glm function also produced warnings related to separation.
#
# Further inspection of the joint distribution of 'al' and the target variable showed
# that when al = 0, the outcome was almost always target = 0 (notCKD), suggesting
# quasi-complete separation.
#
# Therefore, we removed 'al' from the model and refitted the logistic regression
# using the remaining three predictors (bgr, hemo, sg), which resulted in a stable 
# and well-estimated final model.

###############################################################
# Step 3-7: Diagnose Separation Problem for Albumin (al)
###############################################################

# Cross-tabulation between albumin (al) and target
sep_table <- table(df_reduced$al, df_reduced$target)
cat("\n--- Cross Table: al vs target ---\n")
print(sep_table)

al_ckd_prop <- prop.table(sep_table, margin = 1)[, "1"]  # CKD 비율만
barplot(
  al_ckd_prop,
  main = "P(CKD | albumin level)",
  xlab = "Albumin (al)",
  ylab = "Proportion of CKD",
  ylim = c(0, 1)
)


# Optional: row-wise proportions to see class distribution by al level
cat("\n--- Row-wise Proportions (P(target | al)) ---\n")
print(prop.table(sep_table, margin = 1))

# Visualization: albumin vs target
library(ggplot2)
ggplot(df_reduced, aes(x = al, y = target)) +
  geom_jitter(height = 0.05, width = 0.05, alpha = 0.5) +
  labs(
    title = "Albumin (al) vs Target: Checking for Separation",
    x = "Albumin (al)",
    y = "Target (0 = notckd, 1 = ckd)"
  ) +
  theme_minimal()

#True AIC profile plot
aic_values <- step_model$anova$AIC
plot(aic_values, type="b", main="AIC Profile Across Candidate Models")
points(which.min(aic_values), min(aic_values),
       col="red", pch=19, cex=1.5)
text(which.min(aic_values), min(aic_values),
     labels="Lowest AIC", pos=4)
abline(v=which.min(aic_values), lty=2, col="red")


###############################################################
# Interpretation (for the report / slides)
#
# - Stepwise AIC initially selected four predictors:
#   bgr, hemo, al, and sg.
#
# - However, the cross-table and the scatter plot above show
#   that albumin (al) almost perfectly separates the outcome:
#
#     * For al = 0, both target = 0 (notckd) and target = 1 (ckd)
#       are observed.
#     * For al = 1, 2, 3, 4, 5, no observations with target = 0
#       appear; all patients in these al categories are ckd (target = 1).
#
# - This pattern corresponds to (quasi-)complete separation in
#   logistic regression: the predictor al can almost perfectly
#   distinguish between ckd and notckd.
#
# - In this situation, glm tries to push the coefficient for al
#   toward ±∞, which leads to:
#     * extremely large standard errors for al,
#     * z-statistics close to 0,
#     * p-values ≈ 0.99,
#     * and repeated warnings such as
#       "glm.fit: fitted probabilities numerically 0 or 1 occurred".
#
# - Therefore, although stepwise AIC selected al, the variable
#   causes a separation problem and prevents reliable maximum
#   likelihood estimation.
#
# - To obtain stable coefficient estimates and valid inference,
#   we exclude al from the final logistic regression model and
#   refit the model using the remaining three predictors:
#   bgr, hemo, and sg.
###############################################################

###############################################################
# =============================================================
# 3-8) Final Logistic Regression Model (Removing 'al')
# =============================================================
###############################################################

# Remove the problematic variable 'al'
final_vars_clean <- setdiff(final_vars, "al")
cat("\n--- Final Variables After Removing 'al' ---\n")
print(final_vars_clean)

# Build updated model formula
final_formula <- as.formula(
  paste("target ~", paste(final_vars_clean, collapse = " + "))
)
cat("\n--- Final Model Formula ---\n")
print(final_formula)

# Fit the final 3-variable logistic regression model
final_glm_full <- glm(
  final_formula,
  data = df_reduced,
  family = binomial
)

cat("\n--- Final Logistic Regression Summary (3-variable model) ---\n")
summary(final_glm_full)

###############################################################
# Interpretation:
#
# After confirming that 'al' causes quasi-complete separation,
# we removed it and refitted the model using the remaining
# predictors (bgr, hemo, sg).
#
# This 3-variable model converges normally, produces stable
# standard errors, significant p-values, and allows for valid
# inference — unlike the 4-variable model that included 'al'.
###############################################################

###############################################################
# 3-9) Odds Ratio (OR) Analysis for Final 3-Variable Model
###############################################################

# Extract coefficients (excluding intercept)
coef_est <- coef(final_glm_full)
coef_est <- coef_est[-1]

# Build OR table
or_table <- data.frame(
  Variable    = names(coef_est),
  Coefficient = as.numeric(coef_est),
  Odds_Ratio  = exp(coef_est)
)

# 95% Confidence Intervals for OR
conf_int <- suppressMessages(confint(final_glm_full))
conf_int <- conf_int[-1, , drop = FALSE]
or_ci <- exp(conf_int)

or_table$OR_2.5  <- or_ci[, 1]
or_table$OR_97.5 <- or_ci[, 2]

cat("\n--- Odds Ratios (with 95% CI) ---\n")
print(or_table)
###############################################################
# Interpretation of Odds Ratios (Final 3-variable Model)
#
# bgr:
#   - OR = 1.04
#   - A one-unit increase in random blood glucose increases CKD odds by ~4%.
#   - 95% CI > 1 → significant positive risk factor.
#
# hemo:
#   - OR ≈ 0.098
#   - A one-unit increase in hemoglobin reduces CKD odds by ~90%.
#   - Strong protective effect and clinically consistent (CKD patients often show anemia).
#
# sg:
#   - OR extremely close to 0 (CI also near zero).
#   - Indicates a very strong negative association with CKD.
#   - Even small increases in specific gravity sharply decrease CKD risk.
###############################################################

###############################################################
# 3-10) Train/Test Split and Confusion Matrix Evaluation
###############################################################

set.seed(572)
train_idx <- createDataPartition(df_reduced$target, p = 0.8, list = FALSE)

train_data <- df_reduced[train_idx, ]
test_data  <- df_reduced[-train_idx, ]

cat("\nTrain size:", nrow(train_data), " | Test size:", nrow(test_data), "\n")

# Fit model on training data
glm_train <- glm(
  final_formula,
  data   = train_data,
  family = binomial
)

# Predict probabilities on test data
test_prob <- predict(glm_train, newdata = test_data, type = "response")

# Convert to class labels using 0.5 cutoff
test_pred_class <- ifelse(test_prob >= 0.5, 1, 0)

# Convert to factors for confusionMatrix
ref_factor  <- factor(test_data$target,      levels = c(0, 1), labels = c("notckd", "ckd"))
pred_factor <- factor(test_pred_class,       levels = c(0, 1), labels = c("notckd", "ckd"))

cat("\n--- Confusion Matrix (Test Set, cutoff = 0.5) ---\n")
cm <- confusionMatrix(pred_factor, ref_factor, positive = "ckd")
print(cm)
###############################################################
# Interpretation of Confusion Matrix (Test Set)
#
# Accuracy = 0.9625
#   - The model achieves very high predictive performance overall.
#
# Sensitivity = 0.9608
#   - Correctly identifies ~96% of CKD patients.
#
# Specificity = 0.9655
#   - Correctly identifies ~96.5% of non-CKD patients.
#
# PPV = 0.98
#   - When the model predicts CKD, it is correct 98% of the time.
#
# NPV = 0.933
#   - When predicting non-CKD, about 93% are truly non-CKD.
#
# Overall:
#   - The 3-variable model (bgr, hemo, sg) is stable, highly accurate,
#     and maintains balanced sensitivity and specificity.
#   - Removing 'al' resolved the separation issue and resulted in a
#     clean, well-performing logistic regression model.
###############################################################

###############################################################
# 3-11) ROC Curve and AUC for Final 3-Variable Model
###############################################################

roc_obj <- roc(test_data$target, test_prob)

cat("\n--- ROC / AUC ---\n")
cat("AUC:", as.numeric(auc(roc_obj)), "\n")

plot(
  roc_obj,
  col = "blue",
  lwd = 2,
  main = "ROC Curve – Final 3-variable Logistic Model"
)
abline(a = 0, b = 1, lty = 2, col = "gray")

###############################################################
# 3-12) Final Performance Summary (Test Set)
###############################################################

cat("\n====================================================\n")
cat(" Final 3-variable Logistic Model Summary\n")
cat("----------------------------------------------------\n")
cat("Selected Predictors:", paste(final_vars_clean, collapse = ", "), "\n")
cat("Test Accuracy :", round(cm$overall["Accuracy"], 3), "\n")
cat("Test Sensitivity (Recall for CKD) :", round(cm$byClass["Sensitivity"], 3), "\n")
cat("Test Specificity (Recall for notCKD):", round(cm$byClass["Specificity"], 3), "\n")
cat("Test AUC      :", round(as.numeric(auc(roc_obj)), 3), "\n")
cat("====================================================\n")

###############################################################
# Step 4-1: H1 Robustness Check — MCAR / MNAR Simulation
# Objective: Evaluate whether the H1 conclusion (bp ~ appetite)
#            remains consistent under different missing data mechanisms.
###############################################################

library(VIM)
library(caret)
library(dplyr)

###############################################################
# Step 4-1-1: Prepare Base Dataset for H1 Simulation
###############################################################

df_h1 <- df2[, c("bp", "appet")]
df_h1 <- df_h1[!is.na(df_h1$appet), ]   # Remove rows with missing appetite

###############################################################
# Step 4-1-2: Helper Function — Run t-test After KNN Imputation
###############################################################

run_ttest <- function(df) {
  
  # Convert appetite into numeric for KNN (good=0, poor=1)
  df$appet_num <- ifelse(df$appet == "good", 0, 1)
  
  # Apply KNN imputation
  df_imp <- VIM::kNN(df, k = 5, imp_var = FALSE)
  
  # Restore appetite factor
  df_imp$appet_imp <- ifelse(df_imp$appet_num >= 0.5, "poor", "good")
  
  # Group separation
  bp_good <- df_imp$bp[df_imp$appet_imp == "good"]
  bp_poor <- df_imp$bp[df_imp$appet_imp == "poor"]
  
  # Perform t-test
  t_result <- t.test(bp_good, bp_poor, var.equal = FALSE)
  
  return(list(
    t = as.numeric(t_result$statistic),
    p = as.numeric(t_result$p.value),
    mean_good = mean(bp_good),
    mean_poor = mean(bp_poor),
    decision = ifelse(t_result$p.value < 0.05, "Reject H0", "Fail to Reject H0")
  ))
}

###############################################################
# Step 4-1-3: MCAR Injection Function (10–50%)
###############################################################

inject_mcar <- function(df, rate, seed = 572) {
  set.seed(seed)
  out <- df
  
  # MCAR in bp
  idx_bp <- which(!is.na(out$bp))
  n_bp <- floor(length(idx_bp) * rate)
  out$bp[sample(idx_bp, n_bp)] <- NA
  
  # MCAR in appetite
  idx_app <- which(!is.na(out$appet))
  n_app <- floor(length(idx_app) * rate)
  out$appet[sample(idx_app, n_app)] <- NA
  
  return(out)
}

###############################################################
# Step 4-1-4: MNAR Injection Function
# MNAR logic:
#   - If bp > 80 → more likely missing
#   - If appetite = good → more likely missing
###############################################################

inject_mnar <- function(df, rate, seed = 572) {
  set.seed(seed)
  out <- df
  
  hi_bp <- which(out$bp > 80 & !is.na(out$bp))
  miss_bp <- hi_bp[runif(length(hi_bp)) < rate]
  out$bp[miss_bp] <- NA
  
  good_idx <- which(out$appet == "good")
  miss_app <- good_idx[runif(length(good_idx)) < rate]
  out$appet[miss_app] <- NA
  
  return(out)
}

###############################################################
# Step 4-1-5: Run MCAR / MNAR Simulations
###############################################################

results <- list()

# Original
orig <- run_ttest(df_h1)
results[[1]] <- data.frame(
  Type = "Original",
  t = orig$t,
  p = orig$p,
  Mean_good = orig$mean_good,
  Mean_poor = orig$mean_poor,
  Decision = orig$decision
)

# MCAR (10–50%)
rates <- c(0.1, 0.2, 0.3, 0.4, 0.5)
for (r in rates) {
  df_mcar <- inject_mcar(df_h1, r)
  res <- run_ttest(df_mcar)
  results[[length(results) + 1]] <- data.frame(
    Type = paste0("MCAR ", r*100, "%"),
    t = res$t,
    p = res$p,
    Mean_good = res$mean_good,
    Mean_poor = res$mean_poor,
    Decision = res$decision
  )
}

# MNAR (40%)
df_mnar <- inject_mnar(df_h1, 0.4)
res_mnar <- run_ttest(df_mnar)
results[[length(results) + 1]] <- data.frame(
  Type = "MNAR 40%",
  t = res_mnar$t,
  p = res_mnar$p,
  Mean_good = res_mnar$mean_good,
  Mean_poor = res_mnar$mean_poor,
  Decision = res_mnar$decision
)

###############################################################
# Step 4-1-6: Final Summary Table
###############################################################

h1_missing_summary <- do.call(rbind, results)

cat("\n=== Step 4-1 Result: H1 under MCAR/MNAR ===\n")
print(h1_missing_summary)
# Interpretation of MCAR/MNAR Simulation Results (H1: bp ~ appetite)
#
# - MCAR 10–20%: The hypothesis test result remains significant (Reject H0).
#   → The mean BP difference between appetite groups is robust under low missingness.
#
# - MCAR 30–50%: p-values become non-significant (Fail to Reject H0).
#   → High MCAR missing rates reduce statistical power and weaken the conclusion.
#
# - MNAR 40%: The test becomes significant again (Reject H0).
#   → Directional missingness (good appetite more missing) preserves the mean difference.
#
# Overall:
# - H1 conclusion is stable and robust when missingness is low (≤ 20% MCAR).
# - When missingness becomes large (≥ 30% MCAR), inference becomes unreliable.
# - MNAR patterns can either weaken or preserve significance depending on direction.

###############################################################
# Step 4-2: H2 Robustness Check — MCAR / MNAR Simulation
# Final Model Variables = bgr, hemo, sg
###############################################################

library(VIM)
library(caret)
library(pROC)
library(dplyr)

###############################################################
# Step 4-2-1: Prepare Base Dataset (3 predictors + target)
###############################################################

final_vars_clean <- c("bgr", "hemo", "sg")
X_h2 <- df_reduced[, final_vars_clean]
y_h2 <- df_reduced$target

df_h2_base <- cbind(X_h2, target = y_h2)

###############################################################
# Step 4-2-2: Logistic Regression Evaluation Function
###############################################################

evaluate_logit <- function(df) {
  model <- glm(target ~ ., data = df, family = binomial)
  
  prob <- predict(model, type = "response")
  pred <- ifelse(prob >= 0.5, 1, 0)
  
  acc <- mean(pred == df$target)
  auc_val <- as.numeric(auc(df$target, prob))
  
  cm <- table(df$target, pred)
  tn <- cm[1,1]; fp <- cm[1,2]
  fn <- cm[2,1]; tp <- cm[2,2]
  
  sens <- tp / (tp + fn)
  spec <- tn / (tn + fp)
  
  llf <- logLik(model)[1]
  null_llf <- logLik(glm(target ~ 1, data = df, family = binomial))[1]
  
  pseudoR2 <- 1 - (llf / null_llf)
  
  return(list(
    acc = acc,
    auc = auc_val,
    sens = sens,
    spec = spec,
    pseudoR2 = pseudoR2,
    llf = llf
  ))
}

###############################################################
# Step 4-2-3: MCAR Missingness Injection (Random Missing)
###############################################################

inject_mcar_h2 <- function(df, rate, seed = 572) {
  set.seed(seed)
  out <- df
  mask <- matrix(runif(nrow(out) * ncol(out)) < rate,
                 nrow = nrow(out), ncol = ncol(out))
  out[mask] <- NA
  return(out)
}

###############################################################
# Step 4-2-4: MNAR Missingness Injection (Value-dependent Missing)
###############################################################
# Rule: values above mean → higher probability of being missing

inject_mnar_h2 <- function(df, rate, seed = 572) {
  set.seed(seed)
  out <- df
  
  for (var in c("bgr", "hemo", "sg")) {   # <-- 3 variable version
    thr <- mean(out[[var]], na.rm = TRUE)
    idx <- which(out[[var]] > thr)
    miss_idx <- idx[runif(length(idx)) < rate]
    out[[var]][miss_idx] <- NA
  }
  
  return(out)
}

###############################################################
# Step 4-2-5: KNN Imputation + Model Evaluation Wrapper
###############################################################

run_simulation_h2 <- function(df) {
  imp <- VIM::kNN(df, k = 5, imp_var = FALSE)
  res <- evaluate_logit(imp)
  return(res)
}

###############################################################
# Step 4-2-6: Run MCAR & MNAR Simulations
###############################################################

results <- list()

# Original (no missingness)
orig <- run_simulation_h2(df_h2_base)
results[[1]] <- data.frame(
  Type = "Original",
  MissingRate = 0,
  Accuracy = orig$acc,
  AUC = orig$auc,
  Sensitivity = orig$sens,
  Specificity = orig$spec,
  PseudoR2 = orig$pseudoR2,
  LogLik = orig$llf
)

# MCAR 10–50%
rates <- c(0.1, 0.2, 0.3, 0.4, 0.5)

for (r in rates) {
  df_mcar <- inject_mcar_h2(df_h2_base[, final_vars_clean], rate = r)
  df_mcar$target <- y_h2
  
  res <- run_simulation_h2(df_mcar)
  
  results[[length(results)+1]] <- data.frame(
    Type = paste0("MCAR ", r*100, "%"),
    MissingRate = r,
    Accuracy = res$acc,
    AUC = res$auc,
    Sensitivity = res$sens,
    Specificity = res$spec,
    PseudoR2 = res$pseudoR2,
    LogLik = res$llf
  )
}

# MNAR 40%
df_mnar <- inject_mnar_h2(df_h2_base[, final_vars_clean], rate = 0.4)
df_mnar$target <- y_h2

res_mnar <- run_simulation_h2(df_mnar)

results[[length(results)+1]] <- data.frame(
  Type = "MNAR 40%",
  MissingRate = 0.4,
  Accuracy = res_mnar$acc,
  AUC = res_mnar$auc,
  Sensitivity = res_mnar$sens,
  Specificity = res_mnar$spec,
  PseudoR2 = res_mnar$pseudoR2,
  LogLik = res_mnar$llf
)

###############################################################
# Step 4-2-7: Final Summary Table
###############################################################

h2_summary <- do.call(rbind, results)

cat("\n=== Step 4-2 Result: H2 Robustness Check (Final 3-variable model) ===\n")
print(h2_summary)

###############################################################
# Step 4-2-8: Visualization — AUC Across Missingness Levels
###############################################################

library(ggplot2)

ggplot(h2_summary, aes(x = MissingRate, y = AUC, color = Type)) +
  geom_point(size = 3) +
  geom_line(aes(group = Type), size = 1.2) +
  facet_wrap(~Type) +
  theme_minimal(base_size = 14) +
  labs(title = "AUC under MCAR/MNAR – Final 3-variable Logistic Model")
# ------------------------------------------------------------
# Interpretation of Step 4-2 (Final 3-variable model robustness)
#
# - The logistic regression model using bgr, hemo, sg remains
#   extremely stable under both MCAR and MNAR missingness.
#
# - AUC stays between 0.995 and 1.00 across all missing rates
#   (10%, 20%, 30%, 40%, 50%) and MNAR 40%.
#
# - Accuracy also remains high (0.97–1.00), showing that
#   prediction performance does not collapse even with 50%
#   missingness.
#
# - Sensitivity and specificity remain well-balanced, each above
#   0.96 in all scenarios.
#
# - Pseudo R² remains above 0.90 in all simulations, indicating
#   stable model fit despite missing values.
#
# → Conclusion:
#   The final 3-variable logistic regression model (bgr, hemo, sg)
#   demonstrates very strong robustness. Its performance is nearly
#   unaffected by both MCAR and MNAR missingness, indicating that
#   the model structure is stable and insensitive to missing data
#   patterns.
# ------------------------------------------------------------
