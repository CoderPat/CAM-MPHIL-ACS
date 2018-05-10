DECL_FILE = "../raw_data/V2/parallel/parallel_decl"
BODIES_FILE = "../raw_data/V2/parallel/parallel_bodies"
DESC_FILE = "../raw_data/V2/parallel/parallel_desc"

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
            desc = desc.decode('utf-8')
            if desc not in unique_desc:
                new_bodies.append(body)
                new_decls.append(new_decls)
                new_descs.append(desc)

                unique_desc.add(desc)
        except UnicodeDecodeError:
            pass


    print(len(new_descs))
