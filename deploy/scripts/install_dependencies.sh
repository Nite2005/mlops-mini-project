
# Ensure the script runs in non-interactive mode
export DEBIAN_FRONTEND=noninteractive

# update the package lists
sudo apt-get update -y

# install docker
sudo apt-get install -y docker.io

#start and enable docker
sudo systemctl start docker

sudo sytemctl enable docker 

#install necessary utilities
sudo apt-get install -y unzip curl

# download and install awscli
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/home/ubuntu/awscliv2.zip"
unzip -o /home/ubuntu/awscliv2.zip -d /home/ubuntu
sudo /home/ubuntu/aws/install 

# add ubuntu uesr to docker group to run docker commands without sudo
sudo usermod -aG docker ubuntu

# clean up the aws cli installation files
rm -rf /home/ubuntu/awscliv2.zip /home/ubuntu/aws