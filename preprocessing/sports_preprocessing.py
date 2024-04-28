import pandas as pd

DATA_PATH = 'data/sports.csv'

sports = pd.read_csv(DATA_PATH)

sports['date'] = pd.to_datetime(sports[['year', 'month', 'day']])
sports['weekday'] = sports['date'].dt.day_name().str.lower()

sports.drop(['year', 'month', 'day'], axis=1, inplace=True)

colunms_to_drop = ['home', 'away', 'location', 'weather', 'degree']
sports.dropna(subset=colunms_to_drop, inplace=True)

sports['spectator'] = sports['spectator'].astype(int)

sports['degree'] = sports['degree'].str.replace('℃', '').astype(float)

sports['region'] = sports.location.str[:2]

sports['region'] = sports['region'].replace('의정', '의정부')
sports['region'] = sports['region'].replace('탄천', '용인')
sports['region'] = sports['region'].replace('이순', '아산')
sports['region'] = sports['region'].replace('페퍼', '광주')
sports['region'] = sports['region'].replace('하나', '부천')
sports['region'] = sports['region'].replace('BN', '부산')
sports['region'] = sports['region'].replace('DG', '대구')

sports.drop(columns=['location'], axis=1, inplace=True)

region_translation = {
    '부산': 'Busan',
    '수원': 'Suwon',
    '울산': 'Ulsan',
    '고양': 'Goyang',
    '대구': 'Daegu',
    '잠실': 'Jamsil',
    '창원': 'Changwon',
    '원주': 'Wonju',
    '안양': 'Anyang',
    '대전': 'Daejeon',
    '광주': 'Gwangju',
    '의정부': 'Uijeongbu',
    '화성': 'Hwaseong',
    '천안': 'Cheonan',
    '서울': 'Seoul',
    '인천': 'Incheon',
    '김천': 'Gimcheon',
    '안산': 'Ansan',
    '강릉': 'Gangneung',
    '김포': 'Gimpo',
    '용인': 'Yongin',
    '부천': 'Bucheon',
    '전주': 'Jeonju',
    '포항': 'Pohang',
    '제주': 'Jeju',
    '광양': 'Gwangyang',
    '목동': 'Mokdong',
    '아산': 'Asan',
    '춘천': 'Chuncheon',
    '청주': 'Cheongju',
    '군산': 'Gunsan',
    '진주': 'Jinju',
    '밀양': 'Milyang',
    '상주': 'Sangju',
    '마산': 'Masan',
    '양산': 'Yangsan',
    '순천': 'Suncheon',
    '김해': 'Gimhae',
    '평창': 'Pyeongchang',
    '충주': 'Chungju',
    '거제': 'Geoje'
}

sports['region'] = sports['region'].map(region_translation)

sports.drop(columns=['league', 'season'], axis=1, inplace=True)

weather_translation = {
    '맑음': 'Clear',
    '구름조금': 'Partly Cloudy',
    '눈': 'Snow',
    '비': 'Rain',
    '흐림': 'Cloudy',
    '구름많음': 'Mostly Cloudy'
}

sports['weather'] = sports['weather'].replace(weather_translation)

sports['home'] = sports['home'].astype('category').cat.codes
sports['away'] = sports['away'].astype('category').cat.codes
sports['region'] = sports['region'].astype('category').cat.codes
sports['weather'] = sports['weather'].astype('category').cat.codes
sports['weekday'] = sports['weekday'].astype('category').cat.codes
sports['organization'] = sports['organization'].astype('category').cat.codes

sports.drop(columns=['date'], axis=1, inplace=True)

sports.to_csv('data/sports_preprocessed.csv', index=False)