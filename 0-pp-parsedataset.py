#!/usr/bin/env python3
import pickle

ATTR_LIST = ['title','abstract','keywords','year','authors','page_start','page_end','venue','n_citation']
NAME_ATTR_LIST = ['name','aff']
VENUE_ATTR_LIST = ['name']

def process_paper(p):
    # Returns None if the paper is missing ANY of the attributes
    if not all(a in p.keys() for a in ATTR_LIST):
        return None
    if not all(p[k] for k in ATTR_LIST):
        return None
    # Affliliation attribute present and not empty
    if not all(k in author.keys() for author in p['authors'] for k in NAME_ATTR_LIST):
        return None
    if not all(author[k] for author in p['authors'] for k in NAME_ATTR_LIST):
        return None
    # Venue Attribute present and not empty
    if not all(a in p['venue'].keys() for a in VENUE_ATTR_LIST):
        return None
    if not all(p['venue'][k] for k in VENUE_ATTR_LIST):
        return None

    # Copies only the attributes in the list from the old paper to the new paper
    new_p = {k:v for (k,v) in p.items() if k in ATTR_LIST and v}
    if not all(a in new_p.keys() for a in ATTR_LIST):
        return None
    else:
        return new_p

def transform(p):
    # Authors
    authors = p['authors']
    author_list = [a['name'].upper() for a in authors]
    p['authors'] = author_list
    # Affiliations
    if not any(not 'aff' in a.keys() for a in authors):
        aff_list = [a['aff'].upper() for a in authors]
        p['aff'] = aff_list
    # Page count
    if p['page_start'].isnumeric() and p['page_end'].isnumeric():
        page_cnt = int(p['page_end']) - int(p['page_start'])
        del p['page_start']
        del p['page_end']
        p['page_count'] = page_cnt + 1
    else:
        p['page_count'] = None
    p['venue'] = p['venue']['name'].upper()

def hist_attr(papers,attr):
    vals = {}

    for p in papers.values():
        if not p[attr] in vals.keys():
            vals[p[attr]] = 1
        else:
            vals[p[attr]] += 1

    for k,v in sorted(vals.items(), key=lambda kv: kv[1],reverse=True):
        print("%s: %d" % (k,v))

if __name__ == "__main__":
    filename = 'data/trajectory_data.list'
    with open(filename, 'rb') as data_source:
        data = pickle.load(data_source)

    papers = {}
    # List of publications that meet the criteria
    for author in data:
        for p in author['pubs']:
            pID = p['id'].upper()
            # Create a deep copy with the specified attributes
            new_p = process_paper(p)
            if new_p and not pID in papers.keys():
                transform(new_p)
                papers[pID] = new_p

    print("Processed: %d papers" % len(papers))
    with open('data/database.pickle', 'wb') as f:
        pickle.dump(papers, f)
