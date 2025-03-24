# TransTEC
import numpy as np
import pandas as pd
from scenario_tt import scenarios
import math
# Monte Carlo 模拟参数
vehicle_type_probs = {'h_trailer_hws': 0, 'h_trailer_hwl': 1}
province_probs_by_vehicle = {
    'h_trailer_hws': {'黑龙江': 0.015, '吉林': 0.015, '辽宁': 0.015, '北京': 0.02, '天津': 0.015, '河北': 0.03, '山西': 0.05, '内蒙古': 0.015, '新疆': 0.015, '甘肃': 0.015, '陕西': 0.015, '宁夏': 0.015, '青海': 0.015, '河南': 0.06, '湖北': 0.02, '湖南': 0.015, '上海': 0.05, '江苏': 0.16, '浙江': 0.09, '山东': 0.07, '安徽': 0.02, '福建': 0.015, '江西': 0.015, '四川': 0.05, '云南': 0.015, '重庆': 0.015, '贵州': 0.015, '西藏': 0.015, '广西': 0.04, '广东': 0.07, '海南': 0.015},
    'h_trailer_hwm': {'黑龙江': 0.015, '吉林': 0.015, '辽宁': 0.015, '北京': 0.02, '天津': 0.015, '河北': 0.03, '山西': 0.05, '内蒙古': 0.015, '新疆': 0.015, '甘肃': 0.015, '陕西': 0.015, '宁夏': 0.015, '青海': 0.015, '河南': 0.06, '湖北': 0.02, '湖南': 0.015, '上海': 0.05, '江苏': 0.16, '浙江': 0.09, '山东': 0.07, '安徽': 0.02, '福建': 0.015, '江西': 0.015, '四川': 0.05, '云南': 0.015, '重庆': 0.015, '贵州': 0.015, '西藏': 0.015, '广西': 0.04, '广东': 0.07, '海南': 0.015},
    'h_trailer_hwl': {'黑龙江': 0.03, '吉林': 0.03, '辽宁': 0.03, '北京': 0.01, '天津': 0.03, '河北': 0.05, '山西': 0.04, '内蒙古': 0.03, '新疆': 0.03, '甘肃': 0.03, '陕西': 0.03, '宁夏': 0.03, '青海': 0.03, '河南': 0.03, '湖北': 0.04, '湖南': 0.03, '上海': 0.04, '江苏': 0.03, '浙江': 0.03, '山东': 0.03, '安徽': 0.03, '福建': 0.03, '江西': 0.03, '四川': 0.04, '云南': 0.03, '重庆': 0.02, '贵州': 0.03, '西藏': 0.02, '广西': 0.03, '广东': 0.08, '海南': 0.03},
}
rg_by_province = {'黑龙江': 1.04, '吉林': 1.03, '辽宁': 1.02, '北京': 1.02, '天津': 1.02, '河北': 1.03, '山西': 1.04, '内蒙古': 1, '新疆': 1, '甘肃': 1, '陕西': 1, '宁夏': 1, '青海': 1, '河南': 1.04, '湖北': 1.04, '湖南': 1, '上海': 1, '江苏': 1.02, '浙江': 1, '山东': 1.03, '安徽': 1, '福建': 1, '江西': 1.02, '四川': 1.15, '云南': 1, '重庆': 1, '贵州': 1, '西藏': 1, '广西': 1.04, '广东': 1.01, '海南': 1.01}
kroad_by_province = {'黑龙江': 1.04, '吉林': 1.03, '辽宁': 1.02, '北京': 1.02, '天津': 1.02, '河北': 1.03, '山西': 1.04, '内蒙古': 1, '新疆': 1, '甘肃': 1, '陕西': 1, '宁夏': 1, '青海': 1, '河南': 1.04, '湖北': 1.04, '湖南': 1, '上海': 1, '江苏': 1, '浙江': 1, '山东': 1.03, '安徽': 1, '福建': 1, '江西': 1.02, '四川': 1.15, '云南': 1, '重庆': 1, '贵州': 1, '西藏': 1, '广西': 1.04, '广东': 1.01, '海南': 1.01}
temperature_by_province = {'黑龙江': 1.3, '吉林': 1.1, '辽宁': 1.1, '北京': 1, '天津': 1, '河北': 1, '山西': 1, '内蒙古': 1.2, '新疆': 1.2, '甘肃': 1, '陕西': 1, '宁夏': 1, '青海': 1, '河南': 1, '湖北': 1, '湖南': 1, '上海': 1, '江苏': 1, '浙江': 1, '山东': 1.03, '安徽': 1, '福建': 1, '江西': 1, '四川': 1, '云南': 1, '重庆': 1, '贵州': 1, '西藏': 1.2, '广西': 1.3, '广东': 1.3, '海南': 1.3}
traffic_by_province = {'黑龙江': 1, '吉林': 1, '辽宁': 1, '北京': 1.2, '天津': 1.2, '河北': 1, '山西': 1, '内蒙古': 1, '新疆': 1, '甘肃': 1, '陕西': 1, '宁夏': 1, '青海': 1, '河南': 1.4, '湖北': 1, '湖南': 1, '上海': 1, '江苏': 1, '浙江': 1, '山东': 1, '安徽': 1, '福建': 1, '江西': 1.02, '四川': 1.15, '云南': 1, '重庆': 1, '贵州': 1, '西藏': 1, '广西': 1, '广东': 1.1, '海南': 1.1}

def monte_carlo_simulation(num_simulations):
    results_by_scenario = {}
    for _ in range(num_simulations):
        vehicle = np.random.choice(list(vehicle_type_probs.keys()), p=list(vehicle_type_probs.values()))
        province = np.random.choice(list(province_probs_by_vehicle[vehicle].keys()),
                                    p=list(province_probs_by_vehicle[vehicle].values()))
        rg = rg_by_province[province]
        kroad = kroad_by_province[province]
        temperature = temperature_by_province[province]
        traffic = traffic_by_province[province]
        scenario = scenarios[vehicle]
        opd = round(np.random.normal(scenario['mean_opd'], scenario['std_opd']))
        if not (1 <= opd <= 365):
            opd = 340
        distance = float(np.random.choice(scenario['distance']))
        frequence = float(np.random.choice(scenario['frequence']))
        loadfactor = np.random.choice(scenario['loadfactor'])
        dvkd = round(distance)*round(frequence)*2
        avkd = opd*round(distance)*round(frequence)*2
        results = {
            'year': year, 'vehicle': vehicle, 'province': province, 'rg': rg, 'kroad': kroad,
            'temperature': temperature, 'traffic': traffic, 'opd': opd, 'distance': distance,
            'frequence': frequence, 'loadfactor': loadfactor, 'dvkd': dvkd, 'avkd': avkd
        }
        for key in results:
            if key not in results_by_scenario:
                results_by_scenario[key] = []
            results_by_scenario[key].append(results[key])

    df = pd.DataFrame(results_by_scenario)
    return df


all_years_df = pd.DataFrame()
# Define the range of years
years = range(2018,2030)

# Path to the Excel file where results will be saved
temp_costoutput_file_path = 'comtt_annual_cost.xlsx'

# Placeholder for all processed data
costoutput_data = []

# Create an ExcelWriter object to write each year's data to a different sheet
with (pd.ExcelWriter(temp_costoutput_file_path, engine='openpyxl', mode='w') as writer):
    for year in years:
        usefactor = 0.8
        shape = (-3.48315166110525E-07)
        fastsim_file_path = 'tt_output_fastsim10.xlsx'
        vehicle_file_path = 'tt_vehicle_represent_params10.xlsx'
        year_file_path = 'year_varies10.xlsx'
        Urea_usage_percent = 0.1# 百公里使用尿素为柴油5%
        unit_urea_usage_cost =3  # 元/L可兰素
        rate_freeway = 250  # 元/100km 高速费
        UF = 0.4
        discount_rate = 0.04
        blendrate = 0.6
        fastsim_batteryprice_df = pd.read_excel(fastsim_file_path, sheet_name='batteryprice', index_col=0, header=0)
        fastsim_bodyprice_df = pd.read_excel(fastsim_file_path, sheet_name='bodyprice', index_col=0, header=0)
        year_fcprice_df = pd.read_excel(year_file_path, sheet_name='fcprice', index_col=0, header=0)
        region_fcprice_df = pd.read_excel(year_file_path, sheet_name='regionfcprice', index_col=0, header=0)
        year_subsidy_df = pd.read_excel(year_file_path, sheet_name='subsidy', index_col=0, header=0)
        region_subsidy_df = pd.read_excel(year_file_path, sheet_name='regionsubsidy', index_col=0, header=0)
        year_ch_efficiency_df = pd.read_excel(year_file_path, sheet_name='ch_efficiency', index_col=0, header=0)
        year_retire_df = pd.read_excel(year_file_path, sheet_name='retire', index_col=0, header=0)
        year_ptretire_df = pd.read_excel(year_file_path, sheet_name='ptretire', index_col=0, header=0)
        year_yearretire_df = pd.read_excel(year_file_path, sheet_name='yearretire', index_col=0, header=0)
        year_charging_prob_factor_df = pd.read_excel(year_file_path, sheet_name='yearchargprob', index_col=0, header=0)
        year_unittonkmcost_df = pd.read_excel(year_file_path, sheet_name='unittonkmcost', index_col=0, header=0)
        fastsim_riskwtp_df = pd.read_excel(fastsim_file_path, sheet_name='risk_wtp', index_col=0, header=0)
        fastsim_batteryrent_df = pd.read_excel(fastsim_file_path, sheet_name='batteryrent', index_col=0, header=0)
        fastsim_FC_df = pd.read_excel(fastsim_file_path, sheet_name='FC', index_col=0, header=0)
        fastsim_yearFC_df = pd.read_excel(fastsim_file_path, sheet_name='yearFC', index_col=0, header=0)
        fastsim_MSRP_df = pd.read_excel(fastsim_file_path, sheet_name='MSRP', index_col=0, header=0)
        fastsim_yearMSRP_df = pd.read_excel(fastsim_file_path, sheet_name='yearMSRP', index_col=0, header=0)
        fastsim_mmcost_df = pd.read_excel(fastsim_file_path, sheet_name='mmcost', index_col=0, header=0)
        fastsim_retirekm_df = pd.read_excel(fastsim_file_path, sheet_name='retirekm', index_col=0, header=0)
        fastsim_retireyear_df = pd.read_excel(fastsim_file_path, sheet_name='retireyear', index_col=0, header=0)
        fastsim_insura_df = pd.read_excel(fastsim_file_path, sheet_name='insura', index_col=0, header=0)
        fastsim_cargomass_df = pd.read_excel(fastsim_file_path, sheet_name='cargomass', index_col=0, header=0)
        fastsim_cargofc_df = pd.read_excel(fastsim_file_path, sheet_name='cargoFC', index_col=0, header=0)
        fastsim_volume_df = pd.read_excel(fastsim_file_path, sheet_name='volume', index_col=0,
                                                    header=0)
        fastsim_battery_capacity_df = pd.read_excel(fastsim_file_path, sheet_name='battery_capacity', index_col=0,
                                                    header=0)
        fastsim_mmavalibility_df = pd.read_excel(fastsim_file_path, sheet_name='mmavalibility', index_col=0, header=0)
        fastsim_unitcostablecapacity_df = pd.read_excel(fastsim_file_path, sheet_name='unitcostablecapacity',
                                                        index_col=0,
                                                        header=0)
        fastsim_unitcostrestcapacity_df = pd.read_excel(fastsim_file_path, sheet_name='unitcostrestcapacity',
                                                        index_col=0,
                                                        header=0)
        fastsim_tonkm_df = pd.read_excel(fastsim_file_path, sheet_name='tonkm', index_col=0, header=0)
        fastsim_totalmass_df = pd.read_excel(fastsim_file_path, sheet_name='totalmass', index_col=0, header=0)
        fastsim_cubeweight_df = pd.read_excel(fastsim_file_path, sheet_name='cubeweight', index_col=0, header=0)
        fastsim_findingtime_df = pd.read_excel(fastsim_file_path, sheet_name='findingtime', index_col=0, header=0)
        fastsim_highwayratio_df = pd.read_excel(fastsim_file_path, sheet_name='highwayratio', index_col=0, header=0)
        salary_df = pd.read_excel(vehicle_file_path, sheet_name='salary', index_col=0, header=0)
        charging_prob_df = pd.read_excel(vehicle_file_path, sheet_name='charging_prob', index_col=0, header=0)
        fastsim_infrastructure_df = pd.read_excel(fastsim_file_path, sheet_name='infrastructure', index_col=0,header=0)
        fastsim_servicecapacity_df = pd.read_excel(fastsim_file_path, sheet_name='servicecapacity', index_col=0,header=0)
        fastsim_visioncarbonintensity_df = pd.read_excel(fastsim_file_path, sheet_name='visioncarbonintensity', index_col=0,header=0)
        # Monte Carlo simulation and fuel consumption calculation
        df = pd.read_excel('tt_mc_results.xlsx')
        fueloutput_data = []
        for index, row in df.iterrows():
            vehicle = row['vehicle']
            province = row['province']
            daily_mileage_demand1 = row['distance'] *row['frequence']
            daily_mileage_demand = daily_mileage_demand1 / 100
            road_condition_factor = row['kroad']
            temperature_factor = row['temperature']
            traffic_factor = row['traffic']
            load_factor = row['loadfactor']
            opd = row['opd']
            daily_mileage_actual = np.random.normal(daily_mileage_demand, 0.1 * daily_mileage_demand, opd)   # 百公里
            annual_mileage_actual = daily_mileage_actual.sum()
            for power_system in fastsim_FC_df.columns:
                riskwtp = fastsim_riskwtp_df.loc[year, power_system]
                unitbatteryrent = fastsim_batteryrent_df.loc[year, power_system]
                riskcost = -riskwtp * math.exp(shape * (-6000000))
                unitbatteryprice = fastsim_batteryprice_df.loc[year, power_system]
                bodyprice = fastsim_bodyprice_df.loc[vehicle, power_system]
                mmavalibility_factor = fastsim_mmavalibility_df.loc[year, power_system]
                unitcost_ablecapacity = fastsim_unitcostablecapacity_df.loc[year, power_system]
                unitcost_restcapacity = fastsim_unitcostrestcapacity_df.loc[year, power_system]
                findingtime = fastsim_findingtime_df.loc[power_system,province]
                if power_system == 'DICE':
                    DICE_findingtime = findingtime
                if power_system == 'BEVF282':
                    BEV_findingtime = findingtime
                if power_system == 'EDICE':
                    EDICE_findingtime = findingtime
                if power_system == 'ELICE':
                    ELICE_findingtime = findingtime
                year_fuel_consumption = fastsim_yearFC_df.loc[year, power_system]
                highwayratio = fastsim_highwayratio_df.loc[power_system,province]
                visioncarbonintensity = fastsim_visioncarbonintensity_df.loc[year, power_system]
                infrastructure = fastsim_infrastructure_df.loc[year, power_system]
                servicecapacity = fastsim_servicecapacity_df.loc[year, power_system]
                infrastructurecost = infrastructure / servicecapacity
                battery_capacity = fastsim_battery_capacity_df.loc[vehicle, power_system]
                volume = fastsim_volume_df.loc[vehicle, power_system]
                retirekm = fastsim_retirekm_df.loc[vehicle, power_system]
                retireyear = fastsim_retireyear_df.loc[vehicle, power_system]
                batteryprice = unitbatteryprice * battery_capacity
                retire_age = retirekm / 100 / annual_mileage_actual
                if retire_age >15:
                    retire_age = 15
                #batteryrentcost = unitbatteryrent * 12 * retire_age
                batteryrentcost = 0
                if power_system == 'DICE':
                    DICE_volume = volume
                if power_system == 'BEVF282':
                    BEVS_battery_capacity = battery_capacity
                unit_mmcost = fastsim_mmcost_df.loc[vehicle, power_system]
                yearMSRP_value = fastsim_yearMSRP_df.loc[year, power_system]
                mmavalibility = -200000 * math.log(mmavalibility_factor, 2.71828)
                MSRP_value = (bodyprice + batteryprice + batteryrentcost)*yearMSRP_value
                purchasetax = MSRP_value/(1+0.17)*0.1
                if power_system in ['FCEV',  'BEVF282','BEVF423','BEVE282', 'BEVF729','BEVE512','BEVU729']:
                    purchasetax = purchasetax * 0
                insura_cost = fastsim_insura_df.loc[vehicle, power_system]
                standardcargomass = 40
                totalmass1 = fastsim_totalmass_df.loc[year, power_system]
                totalmass = round(totalmass1, 1)
                cubeweight = fastsim_cubeweight_df.loc[year, power_system]
                cubeweight = round(cubeweight, 1)
                maxloadmass = round(totalmass - cubeweight, 1)
                if maxloadmass in fastsim_cargofc_df.index and power_system in fastsim_cargofc_df.columns:
                    fullcargofc = fastsim_cargofc_df.loc[maxloadmass, power_system]
                unit_tonkm_cost = year_unittonkmcost_df.loc[year, province]
                half_cargomass = round(maxloadmass / 2, 1)
                if half_cargomass in fastsim_tonkm_df.index and power_system in fastsim_tonkm_df.columns:
                    option_half_tonkm = fastsim_tonkm_df.loc[half_cargomass, power_system]
                realcargomass1 = load_factor * (totalmass - cubeweight)
                realcargomass = round(realcargomass1, 1)
                if realcargomass in fastsim_cargofc_df.index and power_system in fastsim_cargofc_df.columns:
                    real_unit_fc = fastsim_cargofc_df.loc[realcargomass, power_system] / road_condition_factor / temperature_factor / traffic_factor * year_fuel_consumption
                if power_system == 'DICE':
                    DICE_real_unit_fc =real_unit_fc
                if power_system == 'BEVF282':
                    BEV_real_unit_fc =real_unit_fc
                if power_system == 'NGICE':
                    NGICE_real_unit_fc = real_unit_fc

                daily_fuel_consumption = daily_mileage_actual * real_unit_fc
                annual_fuel_consumption = daily_fuel_consumption.sum()  # L kwh

                if power_system == 'DICE':
                    DICE_visioncarbonintensity =visioncarbonintensity
                if power_system == 'BEVF282':
                    BEV_visioncarbonintensity =visioncarbonintensity
                if power_system == 'NGICE':
                    NGICE_visioncarbonintensity =visioncarbonintensity

                if power_system in ['FCEV']:
                    emission = visioncarbonintensity * annual_fuel_consumption
                elif power_system in ['BEVF282','BEVF423','BEVE282', 'BEVF729','BEVE512','BEVU729']:
                    emission = visioncarbonintensity * annual_fuel_consumption
                elif power_system == 'NGICE':
                    emission = visioncarbonintensity * annual_fuel_consumption
                else:
                    emission = visioncarbonintensity * annual_fuel_consumption
                abatement_cost = emission/1000/1000 * 100
                urea_usage_cost = annual_fuel_consumption * Urea_usage_percent * unit_urea_usage_cost

                if power_system in ['BEVF282','BEVF423','BEVE282', 'BEVF729','BEVE512','BEVU729']:
                    unit_range = 0.9 * battery_capacity / real_unit_fc
                else:
                    unit_range = 0.9 * volume / real_unit_fc # 全载重充满电一次能跑多少100公里
                battery_km = 3000 * battery_capacity / real_unit_fc
                num_refuel_value = round(annual_mileage_actual / unit_range)
                option_maxtonkm = unit_range

                if retire_age >= retireyear:
                    resale_value_factor = 0
                else:
                    retire_age = math.ceil(retire_age)
                    if vehicle in year_retire_df.index and retire_age in year_retire_df.columns:
                        resale_value_factor_1 = year_retire_df.loc[vehicle, retire_age]
                        if vehicle in year_ptretire_df.index and power_system in year_ptretire_df.columns:
                            resale_value_factor_2 = resale_value_factor_1 * year_ptretire_df.loc[vehicle, power_system]
                            if year in year_yearretire_df.index and power_system in year_yearretire_df.columns:
                                resale_value_factor = resale_value_factor_2 * year_yearretire_df.loc[
                                   year, power_system]


                if year in year_ch_efficiency_df.index and power_system in year_ch_efficiency_df.columns:
                    ch_efficiency = year_ch_efficiency_df.loc[year, power_system]  # kwh/hour
                    if power_system == 'DICE':
                        DICE_ch_efficiency = ch_efficiency
                    if power_system == 'BEVF282':
                        BEV_ch_efficiency = ch_efficiency
                    if year in year_fcprice_df.index and power_system in year_fcprice_df.columns:
                        regionfcprice = region_fcprice_df.loc[power_system, province]
                        fcprice = year_fcprice_df.loc[year, power_system] * regionfcprice  # $/L  $/kwh
                        if power_system == 'DICE':
                            DICE_fuelprice = fcprice
                        if power_system == 'BEVF282':
                            BEVS_fuelprice = fcprice
                        if power_system == 'NGICE':
                            NGICE_fuelprice = fcprice
                        fuel_cost = annual_fuel_consumption * fcprice
                        if power_system == 'DICE':
                            DICE_fcprice = fcprice
                        if vehicle in salary_df.index and province in salary_df.columns:
                            unit_salary = salary_df.loc[vehicle, province]
                            if power_system in charging_prob_df.index and province in charging_prob_df.columns:
                                charging_prob_now = charging_prob_df.loc[power_system, province]
                                if year in year_charging_prob_factor_df.index and power_system in year_charging_prob_factor_df.columns:
                                    charging_prob_factor = year_charging_prob_factor_df.loc[year, power_system]
                                    charging_prob = charging_prob_now * charging_prob_factor
                                    if power_system == 'DICE':
                                        DICE_charging_prob = charging_prob
                                    if power_system == 'BEVF282':
                                        BEVS_charging_prob = charging_prob

                                    ablecapacity = 0.9 * 0.9 *volume
                                    restcapacity = 0.5 *0.9 * volume
                                    pro_ablecapacity = 0.6
                                    pro_restcapacity = 1 - pro_ablecapacity
                                    final_refuelcapacity = ablecapacity * pro_ablecapacity + restcapacity * pro_restcapacity
                                    charging_hour = 1  # hour/day
                                    actual_fuel_capacity = min(charging_hour * ch_efficiency,
                                                               ablecapacity)
                                    daily_km_access = 0.9 * battery_capacity / real_unit_fc / 100
                                    daily_km_access_array = np.full(daily_mileage_actual.shape, daily_km_access)
                                    daily_mileage_demand_array = np.full(daily_mileage_actual.shape,
                                                                         daily_mileage_demand)

                                    if power_system == 'DICE':
                                        basetonkm = option_maxtonkm
                                        baseloadmass = maxloadmass
                                    if load_factor < 1:
                                        loss_cost2 = 0
                                    else:
                                        if maxloadmass >= baseloadmass:
                                            loss_cost2 = 0
                                        else:
                                            loss_cost2 = (baseloadmass - maxloadmass) * annual_mileage_actual *100 * unit_tonkm_cost*0.1
                                    tonkmloss_cost = loss_cost2
                                    if power_system in ['DICE','NGICE']:
                                        tonkmloss_cost = 0
                                    anxiety_cost = 0
                                    for daily_kmaccess, daily_mileageactual, daily_mileage_demand in zip(
                                            daily_km_access_array, daily_mileage_actual, daily_mileage_demand_array):
                                        if power_system in ['DICE', 'NGICE', 'FCEV']:
                                            anxiety_cost = 0
                                            anxiety_cost1 = 0
                                            inconvenience_cost = annual_fuel_consumption / ch_efficiency*unit_salary + round(annual_fuel_consumption / final_refuelcapacity)*findingtime / charging_prob * unit_salary
                                            inconvenience_cost1 = annual_fuel_consumption / ch_efficiency * unit_salary
                                            inconvenience_cost2 = round(annual_fuel_consumption / final_refuelcapacity) * findingtime / charging_prob * unit_salary
                                        elif power_system in ['BEVF282','BEVF729','BEVU729']:
                                            if daily_mileage_demand >= daily_kmaccess:
                                                anxietyfactor = (daily_mileage_demand-daily_kmaccess)/daily_mileage_demand
                                                anxietyprob = 0.15
                                                anxiety_cost2 =anxietyfactor*anxietyprob*daily_mileageactual * DICE_real_unit_fc * DICE_fcprice +anxietyfactor*anxietyprob*daily_mileageactual * DICE_real_unit_fc / DICE_ch_efficiency/charging_prob * unit_salary
                                                anxiety_cost += anxiety_cost2
                                                anxiety_cost1 += 1
                                                inconvenience_cost = (1-anxietyfactor*anxietyprob) * (num_refuel_value/charging_prob*unit_salary*0.001+num_refuel_value*findingtime*unit_salary/charging_prob)

                                            else:
                                                unexpectedrange = daily_mileageactual - daily_mileage_demand
                                                if daily_kmaccess < daily_mileageactual:
                                                    anxiety_cost = unexpectedrange * real_unit_fc / ch_efficiency  * unit_salary +unexpectedrange * real_unit_fc *fcprice
                                                    anxiety_cost1 += 1
                                                    inconvenience_cost1 = annual_fuel_consumption / ch_efficiency *0.01*unit_salary+num_refuel_value*findingtime*unit_salary/charging_prob
                                                else:
                                                    anxiety_cost = 0
                                                    anxiety_cost1 = 0
                                                    inconvenience_cost = num_refuel_value/charging_prob*unit_salary*0.01+num_refuel_value*findingtime*unit_salary/charging_prob

                                        elif power_system in ['BEVF423']:

                                            if daily_mileage_demand >= daily_kmaccess:
                                                anxietyfactor = (daily_mileage_demand-daily_kmaccess)/daily_mileage_demand
                                                anxietyprob = 0.15
                                                anxiety_cost3 =anxietyfactor*anxietyprob*(daily_mileageactual * DICE_real_unit_fc * DICE_fcprice + anxietyfactor*anxietyprob*daily_mileageactual * DICE_real_unit_fc / DICE_ch_efficiency/charging_prob * unit_salary)
                                                anxiety_cost += anxiety_cost3
                                                inconvenience_cost =(1-anxietyfactor*anxietyprob) *(annual_fuel_consumption / ch_efficiency *0.001*unit_salary+num_refuel_value*findingtime*unit_salary/charging_prob)
                                            else:
                                                unexpectedrange = daily_mileageactual - daily_mileage_demand
                                                if daily_kmaccess < daily_mileageactual:
                                                    anxiety_cost = unexpectedrange * real_unit_fc / ch_efficiency  * unit_salary +unexpectedrange * real_unit_fc *fcprice
                                                    anxiety_cost += anxiety_cost3
                                                    inconvenience_cost = annual_fuel_consumption / ch_efficiency *0.01*unit_salary+num_refuel_value*findingtime*unit_salary/charging_prob
                                                else:
                                                    anxiety_cost = 0
                                                    anxiety_cost += anxiety_cost3
                                                    inconvenience_cost = annual_fuel_consumption / ch_efficiency *0.01*unit_salary+num_refuel_value*findingtime*unit_salary/charging_prob

                                        elif power_system in ['BEVE282','BEVE512']:
                                            ch_num = charging_hour * ch_efficiency  #换电次数
                                            actual_fuel_capacity = ch_num * battery_capacity
                                            daily_kmaccess = actual_fuel_capacity / real_unit_fc
                                            if daily_kmaccess < daily_mileageactual:
                                                anxiety_cost1 = daily_mileageactual * DICE_real_unit_fc * DICE_fcprice + daily_mileageactual * DICE_real_unit_fc / DICE_ch_efficiency * unit_salary
                                                anxiety_cost += anxiety_cost1
                                                fuel_time =charging_hour
                                                fuel_time_cost_daliy = charging_hour * unit_salary*0.01
                                                inconvenience_cost = fuel_time_cost_daliy * opd

                                            else:
                                                anxiety_cost = 0
                                                anxiety_cost1 = 0
                                                fuel_time =daily_mileageactual * real_unit_fc/battery_capacity*1/ch_efficiency
                                                fuel_time_cost_daliy = fuel_time* unit_salary*0.01
                                                inconvenience_cost = fuel_time_cost_daliy * opd


                                    if power_system in ['DICE']:
                                        urea_usage_cost = urea_usage_cost
                                    else:
                                        urea_usage_cost = 0
                                    if power_system in ['DICE', 'NGICE', 'FCEV',  'BEVF282','BEVF423','BEVE282', 'BEVF729','BEVE512','BEVU729']:
                                        fueloutput_data.append({
                                            '车型': vehicle, '年份': year, 'MSRP_value': MSRP_value, '动力系统': power_system,'购置税':purchasetax,
                                            '省份': province, '日里程需求': daily_mileage_demand, '年运输天数': opd,
                                            '日行驶里程': daily_mileage_actual, '一次加油行驶里程': unit_range,'一次循环里程': battery_km,
                                            '年行驶里程': annual_mileage_actual,
                                            '年均装载系数': load_factor, '年燃油消耗': annual_fuel_consumption,
                                            'option_maxtonkm': option_maxtonkm,
                                            '年补能次数': num_refuel_value, '吨公里损失成本': tonkmloss_cost,
                                            '车型可获得成本': mmavalibility, '风险溢价': riskcost,
                                            '单位时薪': unit_salary, '可充容量': ablecapacity,
                                            '补能便利性成本': inconvenience_cost,
                                            'option_half_tonkm': option_half_tonkm,
                                            'fullcargofc': fullcargofc, 'real_unit_fc': real_unit_fc,
                                            'anxiety_cost1': anxiety_cost1, '焦虑成本': anxiety_cost,
                                            '单位维护成本': unit_mmcost,'补能站建设成本': infrastructurecost,
                                            '尿素使用成本': urea_usage_cost,
                                            'realfc': real_unit_fc, 'realmass': realcargomass,'loss_cost2':loss_cost2,
                                            '保险费': insura_cost, '报废年龄': retire_age,'电池价格': batteryprice,'电池租赁费用': batteryrentcost,
                                            '报废里程': retirekm, '报废年限': retireyear, '燃料消耗成本': fuel_cost,
                                            '残值率': resale_value_factor,'高速路程占比':highwayratio, '碳减排支出': abatement_cost
                                        })
                                else:
                                    print(
                                        f"Charging probability data not found for power system {power_system} in province {province}")
                            else:
                                print(f"Salary data not found for vehicle {vehicle} in province {province}")
                        else:
                            print(f"Fuel cost data not found for power system {power_system} in year {year}")

        if not fueloutput_data:
            print("No data was processed. Please check the input file and parameters.")
        else:
            output_df = pd.DataFrame(fueloutput_data)
            temp_fueloutput_file_path = 'comtt_an_fu_con_results.xlsx'
            output_df.to_excel(temp_fueloutput_file_path, index=False)
            print(f"Results have been written to {temp_fueloutput_file_path}")

        # 读取临时文件数据
        fc_excel_data = pd.ExcelFile(temp_fueloutput_file_path)  # 读取临时文件
        fc_all_data = {sheet_name: pd.read_excel(temp_fueloutput_file_path, sheet_name=sheet_name) for sheet_name in
                       fc_excel_data.sheet_names}

        costoutput_data = []
        for sheet_name, df in fc_all_data.items():
            for index, row in df.iterrows():
                vehicle = row['车型'].strip()
                year = row['年份']
                MSRP_value = row['MSRP_value']
                power_system = row['动力系统'].strip()
                province = row['省份'].strip()
                opd = int(row['年运输天数'])
                daily_mileage_demand = float(row['日里程需求'])
                annual_mileage_actual = float(row['年行驶里程'])
                load_factor = float(row['年均装载系数'])
                annual_fuel_consumption = float(row['年燃油消耗'])
                num_refuel_value = float(row['年补能次数'])
                unit_salary = float(row['单位时薪'])
                inconvenience_cost = float(row['补能便利性成本'])
                anxiety_cost = float(row['焦虑成本'])
                unit_mmcost = float(row['单位维护成本'])
                retirekm = float(row['报废里程'])
                retireyear = float(row['报废年限'])
                battery_km = float(row['一次循环里程'])
                urea_usage_cost = float(row['尿素使用成本'])
                insura_cost = float(row['保险费'])
                resale_value_factor = float(row['残值率'])
                retire_age = float(row['报废年龄'])
                fuel_cost = float(row['燃料消耗成本'])
                tonkmloss_cost = row['吨公里损失成本']
                mmavalibility = row['车型可获得成本']
                riskcost = row['风险溢价']
                batteryprice = row['电池价格']
                batteryrentcost = row['电池租赁费用']
                highwayratio =  row['高速路程占比']
                purchasetax = row['购置税']
                abatement_cost = row['碳减排支出']
                infrastructurecost = row['补能站建设成本']
                MSRP = MSRP_value / (1 + 0.17)
                freeway_tolls = annual_mileage_actual * rate_freeway*highwayratio
                if power_system in year_fcprice_df.columns:
                    if power_system == 'DICE':
                        baseMSRP = MSRP
                        baseresalevalue = resale_value_factor * MSRP
                    if power_system in ['DICE', 'NGICE', 'BEVE']:
                        resale_value = resale_value_factor * MSRP  # 不可以中途换车的残值,车身价格和动力系统价格分开
                        damageinsura = 280 + resale_value * 0.01088
                        insura_cost = insura_cost + damageinsura
                    elif power_system in ['BEVF282','BEVF423','BEVE282', 'BEVF729','BEVE512','BEVU729']:
                        resale_value = resale_value_factor * (MSRP - batteryprice)*0.9+ (
                                math.ceil(annual_mileage_actual * retire_age / battery_km) * battery_km - (
                                annual_mileage_actual * retire_age)) / battery_km *  batteryprice
                        a=math.ceil(annual_mileage_actual * retire_age / battery_km)

                        damageinsura = 280 + resale_value * 0.01088
                        insura_cost = insura_cost + damageinsura
                    elif power_system in ['FCEV']:
                        resale_value = resale_value_factor * (MSRP - batteryprice)* 0.9 + (
                                math.ceil(annual_mileage_actual * retire_age / battery_km) * battery_km - (
                                annual_mileage_actual * retire_age)) / battery_km * batteryprice
                        damageinsura = 280 + resale_value * 0.01088
                        insura_cost = insura_cost + damageinsura
                    if power_system in ['DICE' , 'NGICE' ]:
                        mm_cost = annual_mileage_actual * unit_mmcost * load_factor + urea_usage_cost
                        anotherbatterycost =0
                    else:
                        mm_cost = annual_mileage_actual * unit_mmcost * load_factor + urea_usage_cost
                        anotherbatterycost =  (math.ceil(annual_mileage_actual * retire_age / battery_km) - 1) *batteryprice

                    if year in year_subsidy_df.index and power_system in year_subsidy_df.columns:
                        regionsubsidy = region_subsidy_df.loc[power_system, province]
                        subsidy = year_subsidy_df.loc[year, power_system] * regionsubsidy
                    extracost = (500 * 12 + 500 * 12 + 2000) / 10
                    # year = yearnow #同样属性的车主换车根据yearnow数据重新选择

                    total_discounted_cost = 0
                    for age in range(1, int(retire_age) + 1):
                        annual_cost =batteryrentcost +fuel_cost + inconvenience_cost + anxiety_cost + mm_cost + insura_cost + freeway_tolls + tonkmloss_cost+ infrastructurecost + abatement_cost
                        discounted_cost = annual_cost / ((1 + discount_rate) ** age)
                        total_discounted_cost += discounted_cost
                    TCO = MSRP_value +anotherbatterycost+ purchasetax - resale_value - subsidy + mmavalibility + total_discounted_cost
                    costoutput_data.append({
                        '车型': vehicle,
                        '年份': year,
                        '动力系统': power_system,
                        '省份': province,
                        '购置成本': MSRP_value,
                        '购置税': purchasetax,
                        '电池租赁费用': batteryrentcost,
                        '保险费': insura_cost,
                        '年补能次数': num_refuel_value,
                        '年燃油费用': fuel_cost,
                        '焦虑成本': anxiety_cost,
                        '维护费用': mm_cost,
                        '残值': resale_value,
                        '电池更换成本':anotherbatterycost,
                        '补能便利性成本': inconvenience_cost,
                        '高速费': freeway_tolls,
                        '吨公里损失成本': tonkmloss_cost,
                        '车型可获得成本': mmavalibility,
                        '其他成本': extracost,
                        '补贴': subsidy,
                        '补能站建设成本': infrastructurecost,
                        '碳减排支出': abatement_cost,
                        'TCO': TCO,
                        '使用年龄': retire_age,
                        '总折现成本':total_discounted_cost,
                        '年度成本': annual_cost

                    })
        # Convert the processed data for the year to a DataFrame
        output_df = pd.DataFrame(costoutput_data)
        # Write the data for the current year to a new worksheet
        output_df.to_excel(writer, sheet_name=f'{year}', index=False)
        print(f"Results for year {year} have been written to the Excel file.")

