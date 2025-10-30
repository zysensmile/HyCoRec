import pickle
import argparse

from edger import redial_edger, tgredial_edger, opendialkg_edger, durecdial_edger

dataset_edger_map = {
    'redial': redial_edger,
    'tgredial': tgredial_edger,
    'opendialkg': opendialkg_edger,
    'durecdial': durecdial_edger
}

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, help='Dataset name')
    args, _ = parser.parse_known_args()

    # run edger
    dataset = args.dataset
    if dataset not in dataset_edger_map:
        raise ValueError(f"Dataset {dataset} is not supported.")
    
    print(f"Running edger for dataset: {dataset}")
    item_edger, entity_edger, word_edger = dataset_edger_map[dataset]()

    # save edger
    pickle.dump(item_edger, open(f"data/edger/{dataset}/item_edger.pkl", "wb"))
    pickle.dump(entity_edger, open(f"data/edger/{dataset}/entity_edger.pkl", "wb"))
    pickle.dump(word_edger, open(f"data/edger/{dataset}/word_edger.pkl", "wb"))
    print(f"Lengths - Item: {len(item_edger)}, Entity: {len(entity_edger)}, Word: {len(word_edger)}")
    print(f"Edger for dataset {dataset} saved successfully.")