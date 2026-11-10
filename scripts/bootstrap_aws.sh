#!/usr/bin/env bash
# Run with: source scripts/bootstrap_aws.sh
# Creates a private S3 bucket and a least-scope SageMaker execution role if absent.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then echo "Run: source scripts/bootstrap_aws.sh" >&2; exit 1; fi
set -e
export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_DEFAULT_REGION="$AWS_REGION"
export MODEL_VERSION="${MODEL_VERSION:-v2}"
export AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
export MODEL_BUCKET="${MODEL_BUCKET:-resunet-model-${AWS_ACCOUNT_ID}-${AWS_REGION}}"
export SAGEMAKER_ROLE_NAME="${SAGEMAKER_ROLE_NAME:-resunet-sagemaker-execution}"
if ! aws s3api head-bucket --bucket "$MODEL_BUCKET" 2>/dev/null; then
  echo "Creating bucket $MODEL_BUCKET in $AWS_REGION"
  if [[ "$AWS_REGION" == "us-east-1" ]]; then
    aws s3api create-bucket --bucket "$MODEL_BUCKET" --region "$AWS_REGION" >/dev/null
  else
    aws s3api create-bucket --bucket "$MODEL_BUCKET" --region "$AWS_REGION" --create-bucket-configuration "LocationConstraint=$AWS_REGION" >/dev/null
  fi
fi
aws s3api put-public-access-block --bucket "$MODEL_BUCKET" --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
if ! aws iam get-role --role-name "$SAGEMAKER_ROLE_NAME" >/dev/null 2>&1; then
  aws iam create-role --role-name "$SAGEMAKER_ROLE_NAME" --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"sagemaker.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
fi
export EXECUTION_ROLE_ARN="$(aws iam get-role --role-name "$SAGEMAKER_ROLE_NAME" --query Role.Arn --output text)"
# SageMaker needs to read the packaged model; keep access scoped to this bucket/prefix.
aws iam put-role-policy --role-name "$SAGEMAKER_ROLE_NAME" --policy-name ResUNetModelRead --policy-document "$(printf '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["s3:GetObject"],"Resource":"arn:aws:s3:::%s/deployment/resunet/*"}]}' "$MODEL_BUCKET")"
echo "AWS_REGION=$AWS_REGION"
echo "MODEL_BUCKET=$MODEL_BUCKET"
echo "EXECUTION_ROLE_ARN=$EXECUTION_ROLE_ARN"
echo "MODEL_VERSION=$MODEL_VERSION"
echo 'Role propagation can take a few seconds before SageMaker accepts it.'
