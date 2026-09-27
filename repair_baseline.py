import sys
import os

in_path = r"output\matching_results.tsv"
out_dir = r"output\emergency_baseline_clean"
out_path = os.path.join(out_dir, "matching_results.tsv")

os.makedirs(out_dir, exist_ok=True)

original_rows = 0
s1_changes = 0
match_changes = 0

seen_s1 = {}
duplicate_s1_details = []

with open(in_path, "r", encoding="utf-8") as fin:
    header = fin.readline()
    
    for line in fin:
        line = line.rstrip("\n")
        parts = line.split("\t", 1)
        original_rows += 1
        
        orig_s1 = parts[0]
        orig_matches = parts[1] if len(parts) > 1 else ""
        
        s1 = orig_s1.strip()
        if s1 != orig_s1:
            s1_changes += 1
            
        matched = []
        has_match_whitespace_change = False
        if orig_matches.strip():
            tokens = orig_matches.split(",")
            for t in tokens:
                t_strip = t.strip()
                if t_strip:
                    matched.append(t_strip)
                    if t_strip != t:
                        has_match_whitespace_change = True
        
        if has_match_whitespace_change:
            match_changes += 1
            
        if s1 in seen_s1:
            # Collision
            duplicate_s1_details.append({
                "s1": s1,
                "orig_s1": orig_s1,
                "orig_matches": orig_matches,
                "existing_matches": seen_s1[s1]
            })
            existing_tokens = seen_s1[s1].split(",") if seen_s1[s1] else []
            combined = existing_tokens[:]
            for m in matched:
                if m not in combined:
                    combined.append(m)
            seen_s1[s1] = ",".join(combined)
        else:
            seen_s1[s1] = ",".join(matched)

normalized_row_count = len(seen_s1)

with open(out_path, "w", encoding="utf-8") as fout:
    fout.write("source1_entity_id\tmatched_entity_ids\n")
    for s1, m in seen_s1.items():
        fout.write(f"{s1}\t{m}\n")

print(f"Original row count: {original_rows}")
print(f"Normalized row count: {normalized_row_count}")
print(f"Unique S1 count: {len(seen_s1)}")
print(f"Number of duplicates: {len(duplicate_s1_details)}")
if len(duplicate_s1_details) > 0:
    for d in duplicate_s1_details[:10]:
        print(f"  Collision on S1: '{d['s1']}' (Original was '{d['orig_s1']}') | existing: '{d['existing_matches']}' | new: '{d['orig_matches']}'")
print(f"Whitespace fixes S1: {s1_changes}")
print(f"Whitespace fixes Matches (rows affected): {match_changes}")
