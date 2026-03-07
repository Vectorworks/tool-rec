resource "aws_ecr_repository" "mixtral" {
  name                 = "bim-command-rec"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = false
  }
}

resource "aws_ecr_repository" "baseline" {
  name                 = "bim-command-rec-baseline"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = false
  }
}
