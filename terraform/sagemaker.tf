# --- Mixtral Model ---

resource "aws_sagemaker_model" "mixtral" {
  name               = "bim-command-rec"
  execution_role_arn = aws_iam_role.sagemaker_execution.arn

  primary_container {
    image          = "${aws_ecr_repository.mixtral.repository_url}:latest"
    model_data_url = "s3://${var.s3_bucket}/${var.mixtral_model_s3_key}"
  }
}

resource "aws_sagemaker_endpoint_configuration" "mixtral" {
  name = "bim-command-rec"

  production_variants {
    variant_name           = "AllTraffic"
    model_name             = aws_sagemaker_model.mixtral.name
    initial_instance_count = 1
    instance_type          = var.mixtral_instance_type
    container_startup_health_check_timeout_in_seconds = 600
  }
}

resource "aws_sagemaker_endpoint" "mixtral" {
  name                 = "bim-command-rec"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.mixtral.name
}

# --- Baseline Model ---

resource "aws_sagemaker_model" "baseline" {
  name               = "bim-command-rec-baseline"
  execution_role_arn = aws_iam_role.sagemaker_execution.arn

  primary_container {
    image          = "${aws_ecr_repository.baseline.repository_url}:latest"
    model_data_url = "s3://${var.s3_bucket}/${var.baseline_model_s3_key}"
  }
}

resource "aws_sagemaker_endpoint_configuration" "baseline" {
  name = "bim-command-rec-baseline"

  production_variants {
    variant_name           = "AllTraffic"
    model_name             = aws_sagemaker_model.baseline.name
    initial_instance_count = 1
    instance_type          = var.baseline_instance_type
    container_startup_health_check_timeout_in_seconds = 120
  }
}

resource "aws_sagemaker_endpoint" "baseline" {
  name                 = "bim-command-rec-baseline"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.baseline.name
}
