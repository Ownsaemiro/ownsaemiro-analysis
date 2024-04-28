import pandas as pd

DATA_PATH = 'data/concerts.csv'

concerts = pd.read_csv(DATA_PATH)

concerts['장르_티켓판매수'] = concerts['지역_티켓판매수'] * (concerts['장르_티켓판매수 점유율'] / 100)
concerts['장르_티켓판매수'] = concerts['장르_티켓판매수'].astype(int)

concerts.drop(columns=['지역_티켓판매수', '지역_티켓판매수 점유율', '지역_티켓판매액', '지역_티켓판매액 점유율', '지역_개막편수', '지역_상연횟수', '장르_관객점유율', '장르_티켓판매수 점유율', '장르_티켓판매액', '장르_개막편수', '장르_상연횟수'], axis=1, inplace=True)

concerts['날짜'] = pd.to_datetime(concerts['날짜'])
concerts['요일'] = concerts['날짜'].dt.day_name().str.lower()

concerts.columns = ['date', 'region', 'genre', 'ticket', 'weekday']

region_translation = {
    '서울': 'seuol',
    '경기': 'gyeonggi',
    '인천': 'incheon',
    '부산': 'busan',
    '대구': 'daegu',
    '광주': 'gwangju',
    '대전': 'daejeon',
    '울산': 'ulsan',
    '세종': 'sejong',
    '경남': 'gyeongnam',
    '경북': 'gyeongbuk',
    '전남': 'jeonnam',
    '전북': 'jeonbuk',
    '충남': 'chungnam',
    '충북': 'chungbuk',
    '강원': 'gangwon',
    '제주': 'jeju'
}

concerts['region'] = concerts['region'].map(region_translation)

genre_translation = {
    '연극': 'play',
    '뮤지컬' : 'musical',
    '서양음악(클래식)': 'classic',
    '한국음악(국악)': 'traditional',
    '대중음악': 'popular music',
    '무용(서양/한국무용)': 'dance',
    '대중무용': 'popular dance',
    '서커스/마술': 'circus/magic',
    '복합': 'etc'
}

concerts['genre'] = concerts['genre'].map(genre_translation)

concerts['region'] = concerts['region'].astype('category').cat.codes
concerts['genre'] = concerts['genre'].astype('category').cat.codes
concerts['weekday'] = concerts['weekday'].astype('category').cat.codes

concerts.drop(columns=['date'], inplace=True)

concerts.to_csv('data/concerts_preprocessed.csv', index=False)