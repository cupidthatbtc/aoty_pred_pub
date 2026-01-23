# Publication run script
# Runs the complete publication pipeline with default configs.
#
# Example usage:
#   .\scripts\run_publication.ps1
#   .\scripts\run_publication.ps1 -c configs/custom.yaml
#
# Full pipeline example (manual steps):
#   python -m aoty_pred.cli prepare -c configs/base.yaml
#   python -m aoty_pred.cli build-features -c configs/base.yaml
#   python -m aoty_pred.cli train -c configs/base.yaml -c configs/publication.yaml
#   python -m aoty_pred.cli predict -c configs/base.yaml -c configs/publication.yaml
#   python -m aoty_pred.cli publication -c configs/base.yaml -c configs/publication.yaml

# Run the publication pipeline
python -m aoty_pred.pipelines.publication $args
exit $LASTEXITCODE
