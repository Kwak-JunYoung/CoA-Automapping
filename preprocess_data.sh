data=(abs_company_admin comp_company_admin plain_company_admin part_company_admin company_admin)

for d in ${data[@]}; do
    args=(
        --preprocess_type ${d}
    )
    echo `python preprocess_data.py ${args[@]}`
done