# ASCII Christmas Tree (2026)

3D로 회전하는 ASCII 크리스마스 트리 애니메이션입니다.  
웹 브라우저(GitHub Pages)와 Python 데스크톱 앱 두 가지 방식으로 실행할 수 있습니다.

## 주요 연출 및 조작
- **실시간 타임라인**: 눈 내리는 밤하늘과 지면 → 아래에서 위로 자라는 트리 → 금빛 나선 가랜드 → 빨강/금빛 오너먼트 → 170개 꼬마전구 → 6개 선물 상자 → 5각 별 폭발 점등 → `"M E R R Y   C H R I S T M A S   2 0 2 6"` 타이핑 배너
- **조작 키**:
  - `Space`: 일시정지 / 재개
  - `R`: 처음부터 다시 시작
  - `S`: 현재 화면 PNG 스크린샷 저장
  - `Esc`: 종료 (Python 실행 시)

## GitHub Pages로 링크 공유하기
1. 이 폴더(`github_upload`) **안의 파일들**을 GitHub 레포지토리 최상위(root)에 업로드합니다.
2. 레포지토리의 **Settings → Pages** 로 이동합니다.
3. **Build and deployment → Source** 를 `Deploy from a branch` 로 선택하고, Branch를 **`main`** / **`/ (root)`** 로 지정한 뒤 **Save** 합니다.
4. 잠시 후 생성되는 `https://<아이디>.github.io/<레포지토리명>/` 링크로 PC·모바일 어디서나 바로 감상할 수 있습니다.

## Python으로 로컬 실행하기
```bash
pip install -r requirements.txt
python christmas_tree.py
```

## Credit
- Original inspiration & ASCII tree design: `dwcminh.ascii`
- Recreation & web port: `jaeyonni`
