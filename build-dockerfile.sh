podman build \
  --volume /home/student.unimelb.edu.au/bxb1/bxstorage/4180-llm/assets/wheelhouse:/workspace/wheelhouse:ro,Z \
  -f Dockerfile \
  -t ee-llm:latest \
  .