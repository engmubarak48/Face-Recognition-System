# face-recognition-facenet-faiss

Face recognition is very importance in many areas, including surveillance, security and attendance. This repository consists of fully implemented face recognition system that is based on MTCNN(Multi-task Cascaded Convolutional Neural Networks) to detect face in image/video, Facenet(A Unified Embedding for Face Recognition and Clustering) to extract embeddings and faiss (a library for efficient similarity search and clustering of dense vectors) to search for matched image. This repo would be beneficial for those who have more than 1m+ images. Instead of using 1-to-1 search, which is very expensive for big databases, here we utilize faiss to do 1-to-many search. Having a query image/embedding/ID, we can get the matched image in a few milliseconds (very fast). I also store the embedding, Bounding boxes and image paths to microsoft SQL database, if someone would like to query from database instead of FAISS. 


NB: I am currently writing a "How to use" article and will publish it soon under this repo. 

I thank to contributers/outhors for providing the pre-trained models.
