import torch

data_tensor = torch.load("tensor_data_0_BSneighbor_BW0.05_MH1200_PW10.pt", weights_only=False)

#print(type(data_tensor)) #c'est un dictionnaire
#print(data_tensor)


data_metadata = torch.load("metadata_seeding_data_BSneighbor_BW0.05_MH1200_PW10.pt", weights_only=False)

#print(type(data_metadata)) #c'est un dictionnaire également
#print(data_metadata)

# --> {'total_events': 1, 'events_per_file': 1, 'nb_bins': 126, 'orphan_hit_fraction': 0.0, 
# 'eta_range': [-3.0, 3.0], 'tensor_format': 'pt', 
# 'file_paths': ['/seeding_data_BSneighbor_BW0.05_MH1200_PW10/tensor_data_0_BSneighbor_BW0.05_MH1200_PW10.pt'], 
# 'file_event_ranges': [(0, 1)]}



dataset = torch.load("tensor_data_0_BSneighbor_BW0.05_MH1200_PW10.pt", weights_only=False)
#print(type(dataset))
#print(dataset.keys())

#dict_keys(['hits_tensor', 'particles_tensor', 'good_pairs', 'hit_to_particle_tensor', 
# 'padding_mask', 'start_event', 'end_event', 'nb_bins', 'batch_events'])


data_transformer = torch.load("transformer.pt", weights_only=False, map_location=torch.device('cpu')) #données déjà entrainé que l'on veut passer dans validation
#print(type(data_transformer))
#print(data_transformer.keys())
#print(data_transformer)


data_1event = torch.load("tensor_data_0_BSneighbor_BW0.05_MH1200_PW10.pt", weights_only=False)

hits = data_1event["hits_tensor"]
mask = data_1event["padding_mask"]

#print("hits original shape:", hits.shape)
# print("mask shape:", mask.shape)

# # Si hits est [batch, features, hits], on remet [batch, hits, features]
# if hits.shape[1] < hits.shape[2]:
#     print("Transposition de hits...")
#     hits = hits.transpose(1, 2)

# print("hits final shape:", hits.shape)

# out = model.encodeSpacePoint(hits=hits, mask=mask)
# print("out shape:", out.shape)


a = range(2, 2 + 1)
print(list(a))