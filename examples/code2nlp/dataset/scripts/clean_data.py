"""
Cleans the dataset proposed in https://arxiv.org/pdf/1707.02275.pdf
Mostly removes duplicates, non-ascii docstrings, leading and trailing quotes 
and simplifies the docstring so that parameter descriptions don't appear.
Also performs train-valid-test splitting
"""
import random 

DECL_FILE = "../raw_data/V2/parallel/parallel_decl"
BODIES_FILE = "../raw_data/V2/parallel/parallel_bodies"
DESC_FILE = "../raw_data/V2/parallel/parallel_desc"

CLEANED_DECL_FILE = "../cleaned_data/parallel_decl"
CLEANED_BODIES_FILE = "../cleaned_data/parallel_bodies"
CLEANED_DECLBODIES_FILE = "../cleaned_data/parallel_declbodies"
CLEANED_DESC_FILE = "../cleaned_data/parallel_desc"

TEST_SIZE = 2000
VALID_SIZE = 2000

#A few tags that normally represent begging of argument descriptions or other information
parameter_taggers = [":param", "Args:", "Atributes:", "Reference:"]

def partition(iter, sizes):
    i = 0
    partitions = ()
    for size in sizes:
        partitions = partitions + (iter[i:i+size],)
        i += size
    if i < len(iter):
        partitions = partitions + (iter[i:],)
    
    return partitions
    

if __name__ == "__main__":
    with open(DECL_FILE, 'r') as f:
        decls = f.readlines()
    with open(BODIES_FILE, 'r') as f:
        bodies = f.readlines()
    with open(DESC_FILE, 'rb') as f:
        descs = f.readlines()

    new_bodies, new_decls, new_descs = [], [], []
    unique_desc = set()
    for body, decl, desc in zip(bodies, decls, descs):
        try:
            desc = desc.decode('ascii')
            desc = str.strip(desc, "'\n") + "\n"
            if desc not in unique_desc:
                new_bodies.append(body)
                new_decls.append(decl)

                for tag in parameter_taggers: 
                    if tag in desc:
                        desc = desc[:desc.find(tag)] + '\n'

                new_descs.append(desc)

                unique_desc.add(desc)
        except UnicodeDecodeError:
            pass


    declbodies = [decl[:-1] + " DCNL" + body for decl, body in zip(new_decls, new_bodies)]

    shuffled_indexes = list(range(len(new_descs)))
    random.shuffle(shuffled_indexes)

    splits = partition(shuffled_indexes, [VALID_SIZE, TEST_SIZE])

    for i, split in enumerate(["valid", "test", "train"]):
        with open(CLEANED_DECL_FILE + "." + split, 'w') as f:
            f.writelines([new_decls[j] for j in splits[i]])
        with open(CLEANED_BODIES_FILE + "." + split, 'w') as f:
            f.writelines([new_bodies[j] for j in splits[i]])
        with open(CLEANED_DECLBODIES_FILE + "." + split, 'w') as f:
            f.writelines([declbodies[j] for j in splits[i]])
        with open(CLEANED_DESC_FILE + "." + split, 'w') as f:
            f.writelines([new_descs[j] for j in splits[i]])
    

    

        
