import difflib, io, os, re, sqlite3
from datetime import datetime
import cv2, numpy as np, pandas as pd, streamlit as st
from PIL import Image
try: import pytesseract
except Exception: pytesseract=None
try: import zxingcpp
except Exception: zxingcpp=None

st.set_page_config(page_title='Fleet Scan',page_icon='🚘',layout='centered',initial_sidebar_state='collapsed')
st.markdown('''<style>
.stApp{background:#f3f6fa}.block-container{max-width:760px;padding:.6rem .7rem 5rem}
.hero{background:linear-gradient(135deg,#052c55,#0876cf);color:#fff;padding:14px 16px;border-radius:15px;margin-bottom:10px}.hero h1{font-size:25px;margin:0}.hero p{font-size:13px;margin:3px 0 0;opacity:.9}
.stButton button,.stDownloadButton button{min-height:50px;border-radius:11px;font-weight:750;width:100%}
[data-testid="stCameraInput"] button{min-height:58px}.card{background:#fff;border:1px solid #dce5ef;border-radius:13px;padding:13px;margin:8px 0}.ok{background:#eaf8ef;border-left:6px solid #2f9e44;padding:12px;border-radius:9px}.bad{background:#fff0f0;border-left:6px solid #d93a3a;padding:12px;border-radius:9px}
@media(max-width:640px){.block-container{padding-top:.35rem}.hero h1{font-size:22px}div[data-testid="stHorizontalBlock"]{gap:.45rem}}
</style><div class="hero"><h1>Fleet Scan</h1><p>Photograph a plate or VIN, confirm the match, and save.</p></div>''',unsafe_allow_html=True)

FILES=[('List_Of_Vehicle_Status_new (Hertz).xlsx','Hertz',2),('MOOV vehicle with status 1311.xlsx','Moov',1),('Insurance Data (AFF).xlsx','AFF',3)]
INDEX_FILE='fleet_index.csv'
VIN_NAMES={'VIN','CHASSIS_NO','CHASSIS_NUMBER','VEHICLE_CHASSIS_NUMBER','FRAME_NO','VIN_CHASIS_NO','VIN_CHASSIS_NO'}
PLATE_NAMES={'PLATE','PLATE_NO','PLATE_NUMBER','LICENSE_NO','LICENSE_NUMBER','REGISTRATION_NO','REGISTRATION_NO_'}
ALIASES={'unit':{'UNIT_NO','UNIT_NUMBER','ASSET_NO','ASSET_NUMBER','ASSET_NO_CODE'},'make':{'MAKE','VEHICLE_MAKE','BRAND'},'model':{'MODEL','VEHICLE_MODEL','MODEL_DESCRIPTION'},'year':{'MODEL_YEAR','YEAR','VEHICLE_YEAR'},'branch':{'BRANCH','CURRENT_BRANCH','VEHICLE_BRANCH'},'location':{'LOCATION','CURRENT_LOCATION','YARD','CITY'},'status':{'STATUS','VEHICLE_STATUS','LA_STATUS'},'km':{'KM','KMS','KILOMETRES','KILOMETERS','CURRENT_ODOMETER','MILEAGE'}}
ACTIONS=['Stock Count','Vehicle Received','Vehicle Moved','Inspection Completed','Vehicle Collected','Auction Preparation','Hayaza Check','Other']
DB='fleet_scan.db'
def txt(v):
    if v is None or pd.isna(v): return ''
    x=str(v).strip(); return x[:-2] if x.endswith('.0') and x[:-2].isdigit() else x
def nc(v): return re.sub(r'_+','_',re.sub(r'[^A-Z0-9]+','_',txt(v).upper())).strip('_')
def vin(v): return re.sub(r'[^A-Z0-9]','',txt(v).upper())
def plate(v): return re.sub(r'[^A-Z0-9\u0600-\u06FF]','',txt(v).upper())
def findcol(df,names):
    m={nc(c):c for c in df.columns}
    for n in names:
        if n in m:return m[n]
    for n,c in m.items():
        if any(x in n for x in names):return c

def prepare(df,file,sheet,owner,priority):
    df=df.copy(); df.columns=[txt(c) for c in df.columns]
    vc,pc=findcol(df,VIN_NAMES),findcol(df,PLATE_NAMES)
    if not vc and not pc:return None
    df['_vin']=df[vc].map(vin) if vc else ''; df['_plate']=df[pc].map(plate) if pc else ''
    for k,a in ALIASES.items():
        c=findcol(df,a); df['_'+k]=df[c].map(txt) if c else ''
    df['_owner']=owner;df['_file']=file;df['_sheet']=sheet;df['_priority']=priority
    return df[(df['_vin']!='')|(df['_plate']!='')]
@st.cache_data(show_spinner=False)
def load_fleet_index():
    path=os.path.join(os.path.dirname(os.path.abspath(__file__)),INDEX_FILE)
    if not os.path.exists(path):
        return pd.DataFrame(),[INDEX_FILE]
    df=pd.read_csv(path,dtype=str,keep_default_na=False,encoding='utf-8-sig')
    return df.rename(columns={
        'vin':'_vin','plate':'_plate','unit':'_unit','make':'_make','model':'_model',
        'year':'_year','branch':'_branch','location':'_location','status':'_status','km':'_km',
        'owner':'_owner','source_file':'_file','source_sheet':'_sheet','priority':'_priority'
    }),[]



SCAN_COLUMNS = {
    'id': 'INTEGER PRIMARY KEY',
    'scanned_on': 'TEXT',
    'scanned_by': 'TEXT',
    'action': 'TEXT',
    'scan_type': 'TEXT',
    'scanned_value': 'TEXT',
    'confirmed_vin': 'TEXT',
    'confirmed_plate': 'TEXT',
    'owner': 'TEXT',
    'unit': 'TEXT',
    'make': 'TEXT',
    'model': 'TEXT',
    'year': 'TEXT',
    'branch': 'TEXT',
    'location': 'TEXT',
    'status': 'TEXT',
    'match_method': 'TEXT',
    'source_file': 'TEXT',
    'notes': 'TEXT',
}

def init_db():
    # Streamlit Cloud may retain an older SQLite file from a previous app version.
    # Create the table if missing, then migrate it by adding any new columns.
    with sqlite3.connect(DB, timeout=30) as c:
        c.execute('PRAGMA journal_mode=WAL')
        c.execute('PRAGMA busy_timeout=30000')
        c.execute('CREATE TABLE IF NOT EXISTS scan_log (' + ','.join(
            f'{name} {definition}' for name, definition in SCAN_COLUMNS.items()
        ) + ')')
        existing = {
            row[1] for row in c.execute('PRAGMA table_info(scan_log)').fetchall()
        }
        for name, definition in SCAN_COLUMNS.items():
            if name not in existing and name != 'id':
                c.execute(f'ALTER TABLE scan_log ADD COLUMN {name} {definition}')
        c.commit()


def read_scan_log():
    init_db()
    with sqlite3.connect(DB, timeout=30) as c:
        try:
            return pd.read_sql_query('SELECT * FROM scan_log ORDER BY id DESC', c)
        except Exception:
            # Recover from a stale/incompatible pilot database without crashing the app.
            backup = DB + '.incompatible-' + datetime.now().strftime('%Y%m%d%H%M%S')
            c.close()
            if os.path.exists(DB):
                os.replace(DB, backup)
            init_db()
            with sqlite3.connect(DB, timeout=30) as fresh:
                return pd.read_sql_query('SELECT * FROM scan_log ORDER BY id DESC', fresh)


init_db(); fleet,missing=load_fleet_index()

def rotate(im,a):
    if a==0:return im
    h,w=im.shape[:2];m=cv2.getRotationMatrix2D((w/2,h/2),a,1);co,si=abs(m[0,0]),abs(m[0,1]);nw,nh=int(h*si+w*co),int(h*co+w*si);m[0,2]+=nw/2-w/2;m[1,2]+=nh/2-h/2
    return cv2.warpAffine(im,m,(nw,nh),borderMode=cv2.BORDER_REPLICATE)
def fast_variants(img, kind):
    """Bounded preprocessing for Streamlit Cloud: 4 OCR images, not 30+."""
    rgb=np.array(img.convert('RGB'))
    h,w=rgb.shape[:2]
    # Large phone photos are the main cause of hanging. Cap long edge at 1400px.
    scale=min(1.0,1400.0/max(h,w))
    if scale < 1.0:
        rgb=cv2.resize(rgb,(int(w*scale),int(h*scale)),interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    clahe=cv2.createCLAHE(2.5,(8,8)).apply(gray)
    sharp=cv2.addWeighted(clahe,1.5,cv2.GaussianBlur(clahe,(0,0),1.2),-.5,0)
    binary=cv2.threshold(sharp,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    # VIN labels are often vertical. Only add one 90-degree version.
    return [gray,sharp,binary,rotate(sharp,90)] if kind=='VIN' else [gray,sharp,binary]

def ocr_read(img,kind):
    if not pytesseract:return []
    out=[]
    whitelist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    # One segmentation mode per image keeps scan time bounded.
    config='--oem 3 --psm 11' + (f' -c tessedit_char_whitelist={whitelist}' if kind=='VIN' else '')
    for im in fast_variants(img,kind):
        try:
            data=pytesseract.image_to_data(im,lang='eng',config=config,output_type=pytesseract.Output.DICT,timeout=4)
        except Exception:
            continue
        words=[]
        for text,conf in zip(data.get('text',[]),data.get('conf',[])):
            try: good=float(conf)>=15
            except Exception: good=False
            x=re.sub(r'[^A-Z0-9\u0600-\u06FF]','',str(text).upper())
            if good and x:
                words.append(x)
                if x not in out:out.append(x)
        joined=''.join(words)
        if joined and joined not in out:out.append(joined)
    return out

def direct_candidates(lines,kind):
    out=[]
    for line in lines:
        c=vin(line) if kind=='VIN' else plate(line)
        if kind=='VIN':
            for i in range(max(1,len(c)-16)):
                x=c[i:i+17]
                if len(x)==17 and re.fullmatch(r'[A-HJ-NPR-Z0-9]{17}',x):out.append(x)
        else:
            out += re.findall(r'[A-Z]?\d{2,6}[A-Z]?',c)
    return list(dict.fromkeys(out))
def assisted(lines,kind):
    if fleet.empty:return []
    known=[x for x in fleet['_vin' if kind=='VIN' else '_plate'].dropna().astype(str).unique() if (len(x)==17 if kind=='VIN' else 2<=len(x)<=12)]
    obs=direct_candidates(lines,kind)+[(vin(x) if kind=='VIN' else plate(x)) for x in lines]
    scores={}
    for x in obs:
        if len(x)<(7 if kind=='VIN' else 2):continue
        for k in difflib.get_close_matches(x,known,n=5,cutoff=.62 if kind=='Plate' else .67):scores[k]=max(scores.get(k,0),difflib.SequenceMatcher(None,x,k).ratio())
    return [k for k,_ in sorted(scores.items(),key=lambda z:z[1],reverse=True)[:12]]
def search(q,kind):
    if fleet.empty:return pd.DataFrame(),'',''
    col='_vin' if kind=='VIN' else '_plate';n=vin(q) if kind=='VIN' else plate(q);m=fleet[fleet[col]==n].copy();method='Exact '+kind
    if m.empty and kind=='VIN' and len(n)>=8:m=fleet[fleet[col].str.endswith(n[-8:],na=False)].copy();method='VIN last 8'
    if m.empty and kind=='Plate' and len(n)>=2:m=fleet[fleet[col].str.contains(re.escape(n),na=False)].copy();method='Partial plate'
    if m.empty:return m,'',''
    owners=set(m['_owner']);owner='Moov' if 'Moov' in owners else ('Hertz' if 'Hertz' in owners else 'AFF');return m[m['_owner']==owner].copy(),method,owner

def save(r):
    init_db()
    with sqlite3.connect(DB, timeout=30) as c:
        c.execute('PRAGMA busy_timeout=30000')
        c.execute('INSERT INTO scan_log('+','.join(r)+') VALUES('+','.join('?'*len(r))+')',tuple(r.values()))
        c.commit()

scan_tab,history_tab,status_tab=st.tabs(['📷 Scan','🧾 History','⚙️ Status'])
with scan_tab:
    if missing:st.warning('Some repository files are missing. Check Status.')
    mode=st.segmented_control('Scan type',['Plate','VIN label','VIN barcode / QR','Manual'],default='Plate',selection_mode='single')
    kind='VIN' if mode and mode.startswith('VIN') else 'Plate'
    image_file=None
    if mode!='Manual':
        image_file=st.camera_input('Take photo',key='camera')
        with st.expander('Upload existing photo instead'):
            up=st.file_uploader('Choose image',type=['jpg','jpeg','png','webp'],label_visibility='collapsed');image_file=image_file or up
        st.caption('Keep the camera parallel. Make the plate or VIN fill at least one-third of the image. For glass, try once with flash and once without.')
    if image_file:
        image=Image.open(io.BytesIO(image_file.getvalue())).convert('RGB');st.image(image,width='stretch')
        if st.button('Read photo',type='primary'):
            with st.spinner('Enhancing, rotating, reading and matching against fleet data...'):
                raw=barcode_read(image) if mode=='VIN barcode / QR' else ocr_read(image,kind)
                candidates=list(dict.fromkeys(assisted(raw,kind)+direct_candidates(raw,kind)+raw))[:30]
                st.session_state.candidates=candidates;st.session_state.detected=candidates[0] if candidates else ''
            if not candidates:st.error('No reliable text found. Move closer, clean the glass or plate, and try one photo with flash and one without.')
    cs=st.session_state.get('candidates',[])
    if cs:st.session_state.detected=st.selectbox('Best detected matches',cs)
    value=st.text_input('Confirm or correct '+kind,value=st.session_state.get('detected',''))
    if st.button('Find vehicle',type='primary',disabled=not value.strip()):st.session_state.lookup=(value,kind)
    if 'lookup' in st.session_state:
        q,k=st.session_state.lookup;m,method,owner=search(q,k)
        if m.empty:
            st.markdown('<div class="bad"><b>No fleet match found.</b> Correct the value or save as unmatched.</div>',unsafe_allow_html=True);row={x:'' for x in ['_vin','_plate','_unit','_make','_model','_year','_branch','_location','_status','_file']};owner='Unmatched'
        else:
            labels=[f"{txt(r['_plate']) or '-'} | {txt(r['_vin']) or '-'} | {txt(r['_make'])} {txt(r['_model'])}" for _,r in m.iterrows()];row=m.iloc[labels.index(st.selectbox('Confirm vehicle',labels))]
            st.markdown(f'<div class="ok"><b>{owner} vehicle found</b><br>{txt(row["_make"])} {txt(row["_model"])} · {txt(row["_plate"])} · {txt(row["_vin"])}</div>',unsafe_allow_html=True)
        with st.form('save_form'):
            c1,c2=st.columns(2);user=c1.text_input('Scanned by');action=c2.selectbox('Action',ACTIONS)
            cv=st.text_input('Confirmed VIN',value=txt(row['_vin']) if not m.empty else (vin(q) if k=='VIN' else ''));cp=st.text_input('Confirmed plate',value=txt(row['_plate']) if not m.empty else (q if k=='Plate' else ''))
            loc=st.text_input('Location',value=txt(row['_location']));notes=st.text_area('Notes')
            if st.form_submit_button('Confirm and save',type='primary',disabled=not user.strip()):
                save({'scanned_on':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'scanned_by':user,'action':action,'scan_type':mode,'scanned_value':q,'confirmed_vin':vin(cv),'confirmed_plate':cp,'owner':owner,'unit':txt(row['_unit']),'make':txt(row['_make']),'model':txt(row['_model']),'year':txt(row['_year']),'branch':txt(row['_branch']),'location':loc,'status':txt(row['_status']),'match_method':method,'source_file':txt(row['_file']),'notes':notes});st.success('Saved. Ready for the next vehicle.');st.session_state.pop('lookup',None);st.session_state.candidates=[];st.session_state.detected=''
with history_tab:
    st.caption('History is loaded only when requested so the scanner opens faster.')
    if st.button('Load scan history'):
        st.session_state.show_history=True
    if st.session_state.get('show_history'):
        log=read_scan_log()
        term=st.text_input('Search history')
        if term and not log.empty:
            mask=pd.Series(False,index=log.index)
            for c in log.columns:mask|=log[c].fillna('').astype(str).str.contains(re.escape(term),case=False,na=False)
            log=log[mask]
        st.dataframe(log,width='stretch',hide_index=True)
        if not log.empty:
            out=io.BytesIO()
            with pd.ExcelWriter(out,engine='openpyxl') as w:log.to_excel(w,index=False)
            st.download_button('Download Excel',out.getvalue(),'fleet_scan_log.xlsx',width='stretch')
with status_tab:
    st.subheader('Repository data status')
    st.write(('✅' if INDEX_FILE not in missing else '❌')+f' **Fast fleet index:** {INDEX_FILE}')
    st.metric('Loaded vehicle rows',f'{len(fleet):,}')
    st.caption('Ownership priority: Moov when a VIN exists in both Moov and Hertz; otherwise Hertz or AFF.')
