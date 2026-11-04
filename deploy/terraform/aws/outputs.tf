output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "ecs_cluster" {
  value = aws_ecs_cluster.main.name
}

output "reports_bucket" {
  value = aws_s3_bucket.reports.bucket
}

output "rds_endpoint" {
  value = aws_db_instance.main.address
}
