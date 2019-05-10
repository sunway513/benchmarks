apt update 
apt install -y python3-tk 

pip3 install keras matplotlib opencv-python pillow
wget http://install.aieater.com/dcgan_collapse.tar.gz
tar zxvf dcgan_collapse.tar.gz
cd dcgan_collapse

python3 main_mnist.py