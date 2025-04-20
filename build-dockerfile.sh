sudo podman build \
  --volume /home/student.unimelb.edu.au/bxb1/assets/wheelhouse:/workspace/wheelhouse:ro,Z \
  -f Dockerfile \
  -t ee-llm:latest \
  .
