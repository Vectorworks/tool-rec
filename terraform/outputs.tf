output "mixtral_endpoint_name" {
  description = "Mixtral SageMaker endpoint name"
  value       = aws_sagemaker_endpoint.mixtral.name
}

output "baseline_endpoint_name" {
  description = "Baseline SageMaker endpoint name"
  value       = aws_sagemaker_endpoint.baseline.name
}

output "mixtral_ecr_uri" {
  description = "ECR repository URI for Mixtral container"
  value       = aws_ecr_repository.mixtral.repository_url
}

output "baseline_ecr_uri" {
  description = "ECR repository URI for baseline container"
  value       = aws_ecr_repository.baseline.repository_url
}

output "sagemaker_role_arn" {
  description = "SageMaker execution role ARN"
  value       = aws_iam_role.sagemaker_execution.arn
}
