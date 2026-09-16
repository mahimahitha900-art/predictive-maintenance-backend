from .feature_engineering import derive_values

NAMES = {'TWF':'Tool Wear Failure','HDF':'Heat Dissipation Failure','PWF':'Power Failure','OSF':'Overstrain Failure','RNF':'Random Failure'}

def condition_evidence(row):
    d = derive_values(row)
    limit = {'L':11000, 'M':12000, 'H':13000}[row['product_type']]
    common = {'scope':'AI4I-specific supporting evidence; requires factory calibration', 'proves_physical_cause':False}
    return [
        {**common,'mode':'HDF','triggered': d['temperature_difference'] <= 8.6 and row['rotational_speed'] < 1380, 'rule':'temperature_difference <= 8.6 K AND rotational_speed < 1380 rpm','measured':{'temperature_difference':d['temperature_difference'],'rotational_speed':row['rotational_speed']},'kind':'deterministic_dataset_condition'},
        {**common,'mode':'PWF','triggered':d['mechanical_power_w'] < 3500 or d['mechanical_power_w'] > 9000,'rule':'mechanical_power_w < 3500 W OR > 9000 W','measured':{'mechanical_power_w':d['mechanical_power_w']},'kind':'deterministic_dataset_condition'},
        {**common,'mode':'OSF','triggered':d['overstrain_measure'] > limit,'rule':f'overstrain_measure > {limit} min*Nm for Type {row["product_type"]}','measured':{'overstrain_measure':d['overstrain_measure'],'threshold':limit},'kind':'deterministic_dataset_condition'},
        {**common,'mode':'TWF','triggered':200 <= row['tool_wear'] <= 240,'rule':'Wear falls within AI4I generation interval 200-240 min; replacement/failure is stochastic, not a guaranteed failure cutoff','measured':{'tool_wear':row['tool_wear']},'kind':'stochastic_generation_context'},
        {**common,'mode':'RNF','triggered':False,'rule':'No deterministic sensor cause; sensor readings cannot identify a random event','measured':{},'kind':'unidentifiable_from_sensors'}
    ]
