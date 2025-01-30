library(dplyr)
library(tidyr)
library(lmerTest)
library(lme4)
library(ggplot2)
# Load tibble package
library(tibble)

rm(list=ls())

# Define paths to datasets
datasets <- list(
  Llama8B = "/Github/LLMs_game/Alchemy2/output/data/alchemy2_Llama3_8B_values(not_matched).csv",
  Llama70B = "/Github/LLMs_game/Alchemy2/output/data/alchemy2_Llama3_70B_values(not_matched).csv",
  gpt4o = "/Github/LLMs_game/Alchemy2/output/data/alchemy2_Base_gpt-4o_values(not_matched).csv",
  o1 = "/Github/LLMs_game/Alchemy2/output/data/alchemy2_o1_values(not_matched).csv"
)

# Read and process each dataset to add a unique prefix to the id column
processed_datasets <- lapply(names(datasets), function(dataset_name) {
  df <- read.csv(datasets[[dataset_name]])
  df$id <- paste0(dataset_name, "_", df$id) # Add unique prefix to the id column
  return(df)
})

# Combine all datasets
la2 <- bind_rows(processed_datasets)

# Relabel the id column as standard sequential integers
unique_ids <- unique(la2$id)                      # Get all unique ids
id_mapping <- setNames(seq_along(unique_ids), unique_ids)  # Map unique ids to sequential integers
la2$id <- id_mapping[la2$id]                     # Replace ids with sequential integers


# Create scaled dataframe
la2df <- data.frame(
  trial = scale(la2$trial),
  id = la2$id,
  inventory = scale(la2$inventory),
  cbu = scale(la2$delta_cbu),
  rec = -1 * scale(la2$delta_rec),
  emp = scale(la2$delta_emp),
  bin = scale(la2$delta_bin),
  cbv = scale(la2$delta_cbv),
  truebin = scale(la2$delta_truebin),
  trueemp = scale(la2$delta_trueemp),
  decision = la2$decision,
  model = la2$model,
  temperature = la2$temperature
)

# Fit the model
meus <- glmer(
  decision ~ -1 + emp + cbu + temperature + temperature*emp + temperature*cbu +
    (1 + emp + cbu + temperature + temperature*emp + temperature*cbu | id),
  data = la2df,
  family = "binomial",
  nAGQ = 0,
  control = glmerControl(optimizer = "nloptwrap")
)

# Store results
results <- list()
results$meus <- list(
  model = meus,
  summary = summary(meus)
)

# Extract random effects with conditional variance
ranef_with_var <- ranef(meus, condVar = TRUE)$id

# Convert random effects to dataframe
random_effects_df <- as.data.frame(ranef_with_var) %>%
  tibble::rownames_to_column(var = "id") %>%
  mutate(id = as.integer(id)) %>%  # Convert id to integer
  left_join(la2 %>% select(id, model, temperature) %>% distinct(), by = "id")

# Extract conditional standard errors from the attribute
se_random <- attr(ranef(meus, condVar = TRUE)$id, "postVar")

# Compute standard errors (sqrt of conditional variances)
se_df <- data.frame(
  id = as.integer(rownames(ranef_with_var)),  # Convert rownames to integer
  se_emp = sqrt(se_random[1, 1, ]),          # Std error for emp
  se_cbu = sqrt(se_random[2, 2, ]),          # Std error for cbu
  se_temperature = sqrt(se_random[3, 3, ]),        # Std error for trial
  se_emp_temperature = sqrt(se_random[4, 4, ]),    # Std error for emp:trial
  se_cbu_temperature = sqrt(se_random[5, 5, ])     # Std error for cbu:trial
)

# Merge standard errors back into the random effects dataframe
random_effects_df <- random_effects_df %>%
  left_join(se_df, by = "id")


# Define random effect and standard error columns explicitly
random_effect_columns <- c("emp", "cbu", "temperature.x", "emp:temperature", "cbu:temperature")
se_columns <- c("se_emp", "se_cbu", "se_temperature", "se_emp_temperature", "se_cbu_temperature")

# Compute average random effects and standard errors by model
average_random_effects <- random_effects_df %>%
  group_by(model) %>%
  summarise(
    across(all_of(random_effect_columns), mean, .names = "mean_{.col}"),
    across(all_of(se_columns), mean, .names = "mean_{.col}"),
    .groups = "drop"
  )

# Compute average random effects and standard errors by model and temperature
average_random_effects_by_temp <- random_effects_df %>%
  group_by(model, temperature.y) %>%
  summarise(
    across(all_of(random_effect_columns), mean, .names = "mean_{.col}"),
    across(all_of(se_columns), mean, .names = "mean_{.col}"),
    .groups = "drop"
  )

# Save the results to CSV files
write.csv(average_random_effects, "results\\average_random_effects_by_model_with_se(temp).csv", row.names = FALSE)
write.csv(average_random_effects_by_temp, "results\\average_random_effects_by_model_and_temperature_with_se(temp).csv", row.names = FALSE)

# Extract fixed effects summary (estimates, std errors, z-values, p-values)
fixed_effects_summary <- as.data.frame(coef(summary(meus)))

# Add term names for clarity
fixed_effects_summary$Term <- rownames(fixed_effects_summary)
rownames(fixed_effects_summary) <- NULL

# Reorder columns
fixed_effects_summary <- fixed_effects_summary %>%
  select(Term, Estimate = Estimate, Std_Error = `Std. Error`, Z_Value = `z value`, P_Value = `Pr(>|z|)`)

# Save fixed effects summary into CSV
write.csv(fixed_effects_summary, "results\\fixed_effects_summary(temp).csv", row.names = FALSE)










