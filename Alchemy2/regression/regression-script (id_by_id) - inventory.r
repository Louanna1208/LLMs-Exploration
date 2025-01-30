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
  inventory = la2$inventory,
  cbu = scale(la2$delta_cbu),
  rec = -1 * scale(la2$delta_rec),
  emp = scale(la2$delta_emp),
  bin = scale(la2$delta_bin),
  cbv = scale(la2$delta_cbv),
  truebin = scale(la2$delta_truebin),
  trueemp = scale(la2$delta_trueemp),
  decision = la2$decision,
  model = factor(la2$model),
  temperature = la2$temperature
)
# Set Llama3-8B as the baseline for contrasts
contrasts(la2df$model) <- contr.treatment(levels(la2df$model), base = which(levels(la2df$model) == "Llama3_8B"))

# Fit the model
meus <- glmer(
  inventory ~ -1 + trial*model*temperature +
    (1| id),
  data = la2df,
  family = "poisson",
  nAGQ = 0,
  control = glmerControl(optimizer = "nloptwrap")
)
summary(meus)