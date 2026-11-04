variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "name" {
  type    = string
  default = "arl"
}

variable "image_tag" {
  description = "Image tag in the ECR repository created by this stack. Supply it at apply time."
  type        = string
}

variable "db_password" {
  description = "RDS master password. Do not commit a real value."
  type        = string
  sensitive   = true
}

variable "api_token" {
  description = "Bearer token the Fargate task expects in ARL_API_TOKEN."
  type        = string
  sensitive   = true
  default     = "change-me"
}

variable "budget_email" {
  description = "Address that receives the five-dollar monthly budget alarm."
  type        = string
  default     = "abhay@example.com"
}
