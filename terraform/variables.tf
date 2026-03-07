variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "account_id" {
  description = "AWS account ID"
  type        = string
  default     = "167665646931"
}

variable "s3_bucket" {
  description = "S3 bucket for model artifacts and data"
  type        = string
  default     = "vectorworks-analytics-datalake"
}

variable "mixtral_model_s3_key" {
  description = "S3 key for Mixtral model artifact"
  type        = string
  default     = "tool-rec-data/sagemaker/model.tar.gz"
}

variable "baseline_model_s3_key" {
  description = "S3 key for baseline model artifact"
  type        = string
  default     = "tool-rec-data/sagemaker/baseline_model.tar.gz"
}

variable "mixtral_instance_type" {
  description = "SageMaker instance type for Mixtral endpoint"
  type        = string
  default     = "ml.g4dn.xlarge"
}

variable "baseline_instance_type" {
  description = "SageMaker instance type for baseline endpoint"
  type        = string
  default     = "ml.m5.large"
}

variable "sagemaker_role_name" {
  description = "Name of the SageMaker execution IAM role"
  type        = string
  default     = "tools-rec-iam-role"
}
