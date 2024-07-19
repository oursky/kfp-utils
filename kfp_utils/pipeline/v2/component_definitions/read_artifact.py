from kfp.dsl import Input, Artifact

def read_artifact(artifact: Input[Artifact]) -> str:
    with open(artifact.path, 'r') as artifact_file:
        contents = artifact_file.read()
        return contents
