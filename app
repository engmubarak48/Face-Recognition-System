#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:04:24 2019

@author: mubarak
"""

import pymssql
import numpy as np
import faiss
import os
import tensorflow as tf
from align import detect_face
import facenet
import cv2
from tqdm import tqdm
# some constants kept as default from facenet
minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.85
margin = 44
input_image_size1 = 160
input_image_size2 = 160

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
# read pnet, rnet, onet models from align directory and files are det1.npy, det2.npy, det3.npy
pnet, rnet, onet = detect_face.create_mtcnn(sess, 'align')

## read model
facenet.load_model("Path_To_Model/20180402-114759/20180402-114759.pb")

# Get input and output tensors
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]

class Faiss:
    def __init__(self, directory):

        self.conn = pymssql.connect(server='IP-ADDRESS', user='DB-USER', password='DB-PASSWORD', database='DB-NAME')
        self.cur = self.conn.cursor() 
        self.cur.execute("SELECT COUNT(*) FROM dbo.frdata")
        self.count = self.cur.fetchone()
        if os.path.isfile(directory + '/512_new'):
            self.index = faiss.read_index(directory + '/512_new')
        else:
            print('faiss index is not available')
            pass
    def getFace(self, img):
        faces = []
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, points = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if not len(bounding_boxes) == 0:
            for face in bounding_boxes:
                if face[4] > 0.50:
                    det = np.squeeze(face[0:4])
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0] - margin / 2, 0)
                    bb[1] = np.maximum(det[1] - margin / 2, 0)
                    bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                    bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                    resized = cv2.resize(cropped, (input_image_size1, input_image_size2), interpolation=cv2.INTER_CUBIC)
                    prewhitened = facenet.prewhiten(resized)
                    faces.append(
                        {'face': resized, 'rect': [bb[0], bb[1], bb[2], bb[3]],
                         'embedding': self.getEmbedding(prewhitened)})
        return faces

    def getEmbedding(self, resized):
        reshaped = resized.reshape(-1, input_image_size1, input_image_size2, 3)
        feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
        # print(feed_dict)
        embedding = sess.run(embeddings, feed_dict=feed_dict)
        return embedding
    
    def insertAlldB(self, input_dir):
        count = 0
        for root, directories, filenames in tqdm(os.walk(input_dir)):
            for filename in tqdm(filenames):
                count = count + 1
                image_filename = os.path.join(root, filename)

                try:
                    image = cv2.imread(image_filename)
                    faces = self.getFace(image)
                    for face in faces:
                        new_face_encoding = np.array(face['embedding'])
                        new_face_encoding = new_face_encoding.astype(np.float)
                        face_encoding_string = tuple(new_face_encoding.tolist())
                        data = str(face_encoding_string)
                        data = data.replace('[', '')
                        data = data.replace(']', '')
                        data = data.replace(')', '')
                        data = data[:-1]
                        data = data + ')'
                        bbox = face['rect']

                        sql = "INSERT INTO dbo.frdata (dbo.frdata.filename, dbo.frdata.embeddings, \
                                dbo.frdata.bbox) VALUES ('{}','{}','{}')".format(image_filename, data, bbox)

#                        print(sql)
                        self.cur.execute(sql)
                        self.conn.commit()

                except Exception as e:
                    # pass
                    print("Error: {}".format(e))

    def AddDatafaissAll(self, directory):
        print("Reading features from database...")
        self.cur.execute("SELECT COUNT(*) FROM dbo.frdata")
        count = self.cur.fetchone()

        query = "SELECT embeddings FROM dbo.frdata"
        self.cur.execute(query)   # execute query
        response = self.cur.fetchall()
#        print([response[1]])        
        d = 512
        nb = count[0]
        xb = np.zeros((nb, d)).astype('float32')  # searching set
        index = faiss.IndexFlatL2(d)  # build the index size of vector
        index2 = faiss.IndexIDMap(index)
        ids = np.arange(nb) + 1
        print(index2.is_trained)

        i = 0
        for r in response:
            embedding = r[0]
#            print(embedding)
            embedding = embedding.strip("()")
            embedding = embedding.replace(',', '')
            returned_embedding = np.array([v for v in embedding.split(' ') if v])
            returned_embedding = np.array(returned_embedding)
            returned_embedding = returned_embedding.astype(np.float)
            xb[i] = returned_embedding
            i = i + 1
        # print(xb)

        index2.add_with_ids(xb, ids)  # add vectors to the index
        faiss.write_index(index2, directory + '/512_new')
        print("Completed writing index to file")
        print(index2.ntotal)
        
    def AddDatafaissOne(self, embedding, Id, directory): 
        d = 512
        nb = 1 # nb of queries to add
        xb = np.zeros((nb, d)).astype('float32')  # searching set
        print(embedding)
        embedding = embedding.strip("()")
        embedding = embedding.replace(',', '')
        returned_embedding = np.array([v for v in embedding.split(' ') if v])
        returned_embedding = np.array(returned_embedding)
        returned_embedding = returned_embedding.astype(np.float)
        xb[0] = returned_embedding
        # print(xb)
        print("Adding to index..")
        ids = np.arange(1) + Id
        print(ids)
        self.index.add_with_ids(xb, ids)  # add vectors to the index
        faiss.write_index(self.index, directory + '/512_new')
        print("Completed writing index to file")
        print(self.index.ntotal)

    def search(self, emb):
        d = 512
        nq = 1  # nb of queries
        xq = np.zeros((nq, d)).astype('float32')  # query
        str_emb = emb
        str_emb = np.array(list(str_emb[0])).astype(np.float)
        xq[0] = str_emb
        # print(xq)
        k = 4  # we want to see 4 nearest neighbors
        D, I = self.index.search(xq, k)  # actual search
#        print(I)
        k1, k2, k3, k4 = I[0][0], I[0][1], I[0][2], I[0][3]
        result = str([k1,k2,k3,k4])
        print(k1, k2)
        print("Distance:")
        print(D)
        return result

    def Delete(self, directory, frid):
        print('Make sure you also deleted from Database')
        print(self.index.ntotal)
        self.index.remove_ids(np.arange(1) + int(frid))
        faiss.write_index(self.index, directory + '/512_new')
        print('removed')
        print(self.index.ntotal)


