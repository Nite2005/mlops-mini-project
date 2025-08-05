aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 619071355884.dkr.ecr.ap-south-1.amazonaws.com
docker pull 619071355884.dkr.ecr.ap-south-1.amazonaws.com/emotion_detection:v1
docker stop my-container || true
docker rm my-container || true
docker run -p 80:5000 --name my-app -e DAGSHUB_PAT=5b9f9979416faf718b033e5604938f9647a5b2fa 619071355884.dkr.ecr.ap-south-1.amazonaws.com/emotion_detection:v1