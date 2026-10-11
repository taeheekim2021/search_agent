(() => {
  'use strict';

  const el = (id) => document.getElementById(id);
  const state = {
    key: '',
    connected: false,
    session: 0,
    status: null,
    view: 'overview',
    controllers: new Set(),
    contentLoading: false,
    contentLoaded: false,
    contentOffset: 0,
    contentTotal: 0,
    contentHasMore: false,
    contentRequest: 0,
    contentQuery: '',
    contentPageSize: 20,
    searchPending: false,
    previewPending: false,
    writePending: false,
    preview: null,
    fileRequest: 0,
    searchVerification: null,
    confirmAction: null,
  };
  const views = {
    overview: ['운영 현황', '현재 검색 환경과 콘텐츠 상태를 확인하세요.'],
    contents: ['콘텐츠 관리', '등록 자료를 조회하고 메타데이터 적재를 관리하세요.'],
    search: ['검색 테스트', '검색 조건, 결과의 근거와 실제 모델 실행 여부를 확인하세요.'],
    index: ['인덱스 관리', '설정된 OpenSearch 인덱스를 준비하고 검색에 반영하세요.'],
  };
  const numbers = new Intl.NumberFormat('ko-KR');
  const formatNumber = (value) => (Number.isFinite(value) ? numbers.format(value) : '확인 불가');
  const nowLabel = () =>
    new Date().toLocaleTimeString('ko-KR', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  const listText = (value) => (Array.isArray(value) && value.length ? value.join(', ') : '미지정');
  const node = (tag, text, className) => {
    const item = document.createElement(tag);
    if (text !== undefined && text !== null) item.textContent = String(text);
    if (className) item.className = className;
    return item;
  };
  const tag = (text, tone = 'neutral') => node('span', text, `tag ${tone}`);
  const feedback = (id, text = '', kind = '') => {
    const target = el(id);
    target.textContent = text;
    target.classList.remove('error', 'success', 'warning');
    if (kind) target.classList.add(kind);
  };
  const facts = (id, rows) => {
    const target = el(id);
    target.replaceChildren();
    rows.forEach(([label, value]) => {
      target.append(
        node('dt', label),
        node('dd', value === null || value === undefined || value === '' ? '미확인' : value),
      );
    });
  };
  const safeLink = (url, label) => {
    if (typeof url !== 'string') return null;
    try {
      const parsed = new URL(url);
      if (!['https:', 'http:'].includes(parsed.protocol) || parsed.username || parsed.password)
        return null;
      const link = node('a', label, 'result-source');
      link.href = parsed.href;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      return link;
    } catch {
      return null;
    }
  };
  const ageLabel = (item) =>
    item.min_age === null ||
    item.min_age === undefined ||
    item.max_age === null ||
    item.max_age === undefined
      ? '미상'
      : `${item.min_age}–${item.max_age}세`;

  class ApiError extends Error {
    constructor(message, status, code) {
      super(message);
      this.status = status;
      this.code = code;
    }
  }

  async function api(path, body) {
    const controller = new AbortController();
    const session = state.session;
    state.controllers.add(controller);
    try {
      const headers = { Authorization: `Bearer ${state.key}`, Accept: 'application/json' };
      if (body !== undefined) headers['Content-Type'] = 'application/json';
      const response = await fetch(`/api/admin${path}`, {
        method: body === undefined ? 'GET' : 'POST',
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: controller.signal,
        cache: 'no-store',
        credentials: 'omit',
        mode: 'same-origin',
        redirect: 'error',
      });
      let data;
      try {
        data = await response.json();
      } catch {
        throw new ApiError(
          '서버 응답 형식을 확인할 수 없습니다.',
          response.status,
          'INVALID_RESPONSE',
        );
      }
      if (session !== state.session) throw new DOMException('Session ended', 'AbortError');
      if (!response.ok) {
        const detail = data.detail;
        let message = typeof detail === 'string' ? detail : detail?.message;
        if (Array.isArray(detail?.errors)) {
          const lines = detail.errors
            .slice(0, 5)
            .map(
              (item) =>
                `${Array.isArray(item.loc) ? item.loc.join('.') : '입력값'}: ${item.message || item.type || '유효하지 않음'}`,
            );
          message = [message, ...lines].filter(Boolean).join('\n');
        }
        if (response.status === 401) {
          message = '관리자 키가 올바르지 않습니다. 서버에 설정된 키를 확인해 주세요.';
          if (state.connected) disconnect('관리자 인증에 실패했습니다. 키를 다시 입력해 주세요.');
        } else if (response.status === 404) {
          message =
            message ||
            '관리자 서비스가 비활성 상태입니다. 서버의 SEARCH_ADMIN_API_KEY 설정을 확인해 주세요.';
          if (detail?.code === 'ADMIN_DISABLED' || detail === 'Not Found')
            message =
              '관리자 서비스가 비활성 상태이거나 경로를 찾을 수 없습니다. 서버의 SEARCH_ADMIN_API_KEY 설정을 확인해 주세요.';
        }
        throw new ApiError(
          message || `요청에 실패했습니다. (HTTP ${response.status})`,
          response.status,
          detail?.code,
        );
      }
      return data;
    } finally {
      state.controllers.delete(controller);
    }
  }

  function errorMessage(error, writing = false) {
    if (error instanceof ApiError) return error.message;
    return writing
      ? '작업 결과를 확인하지 못했습니다. 서버에서는 처리가 계속될 수 있습니다. 상태와 콘텐츠를 확인한 후 다시 실행해 주세요.'
      : '서버에 연결하지 못했습니다. 네트워크와 서버 상태를 확인해 주세요.';
  }

  function applyControls() {
    const capabilities = state.status?.capabilities || {};
    const locked =
      state.writePending ||
      state.searchPending ||
      state.previewPending ||
      Boolean(state.status?.operation_in_progress);
    el('disconnect').disabled = state.writePending;
    el('refresh-status').disabled = state.writePending;
    el('run-search').disabled = !state.connected || !capabilities.search || locked;
    el('run-search').textContent = state.searchPending ? '검색 중…' : '검색 실행 ↗';
    el('preview-ingest').disabled = !state.connected || !capabilities.ingest_preview || locked;
    el('preview-ingest').textContent = state.previewPending ? '검증 중…' : '1. 내용 검증';
    el('execute-ingest').disabled =
      !state.connected || !capabilities.ingest_execute || !state.preview || locked;
    el('manifest-json').disabled = state.writePending || state.previewPending;
    el('manifest-file').disabled = state.writePending || state.previewPending;
    el('ensure-index').disabled = !state.connected || !capabilities.index_ensure || locked;
    el('refresh-index').disabled = !state.connected || !capabilities.index_refresh || locked;
    el('content-search-button').disabled =
      !state.connected || !capabilities.browse || state.contentLoading;
    el('content-prev').disabled =
      !state.connected || state.contentLoading || state.contentOffset <= 0;
    el('content-next').disabled = !state.connected || state.contentLoading || !state.contentHasMore;
  }

  function renderStatus(data) {
    state.status = data;
    const sample = Boolean(data.sample);
    const mock = data.mode === 'mock';
    const environment = `${sample ? '샘플 카탈로그' : 'OpenSearch'} · ${mock ? '모의 모드' : '실제 모델 모드'}`;
    el('sidebar-environment').textContent = sample ? '샘플 카탈로그' : 'OpenSearch';
    el('sidebar-note').textContent = mock
      ? '키워드 기반 모의 실행 환경\n실제 AI 모델 미실행'
      : '실제 모델 실행 환경\n추론 결과는 검색 테스트에서 확인';
    el('environment-badge').textContent = environment;
    el('overview-status')
      .closest('.service-banner')
      .classList.toggle('warning', data.status !== 'ok');
    el('overview-status').textContent =
      data.status === 'ok' ? '관리자 서비스에 연결되었습니다' : '운영 상태 확인이 필요합니다';
    el('overview-summary').textContent = data.operation_in_progress
      ? '서버에서 작업을 처리하고 있습니다. 잠시 후 상태를 새로고침해 주세요.'
      : mock
        ? '샘플 콘텐츠로 검색 동작과 관리 흐름을 확인할 수 있습니다.'
        : '현재 서버의 구성과 인덱스 상태를 기준으로 표시합니다.';
    const checked = new Date();
    el('last-checked').dateTime = checked.toISOString();
    el('last-checked').textContent = checked.toLocaleTimeString('ko-KR', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
    el('stat-content').textContent = Number.isFinite(data.content_count)
      ? `${formatNumber(data.content_count)}개`
      : '미확인';
    el('stat-content-note').textContent = sample
      ? '가상 샘플 카탈로그 기준'
      : '설정된 인덱스의 문서 수';
    el('stat-backend').textContent = sample ? 'Sample' : 'OpenSearch';
    el('stat-backend-note').textContent = sample
      ? 'JSON 카탈로그'
      : data.index?.name || '인덱스 확인 필요';
    el('stat-mode').textContent = mock ? '모의 실행' : '실제 모델';
    el('stat-mode-note').textContent = mock
      ? 'Qwen / BGE 추론 미실행'
      : '첫 실행에서 모델 로딩 가능';
    el('stat-model').textContent = state.searchVerification ? '실행 확인' : '미확인';
    el('stat-model-note').textContent = state.searchVerification
      ? `이 창의 검색 성공 · ${state.searchVerification}`
      : '검색 테스트에서 실제 실행 여부 확인';
    el('model-list').replaceChildren();
    const modelStates = {
      disabled: ['비활성', 'neutral'],
      not_loaded: ['로딩 전', 'amber'],
      loaded: ['로딩됨', 'teal'],
      adapter_unverified: ['어댑터 미확인', 'amber'],
    };
    for (const [key, label] of [
      ['embedding', '임베딩 모델'],
      ['reranker', '리랭커 모델'],
    ]) {
      const model = data.models?.[key] || {};
      const [statusLabel, tone] = modelStates[model.state] || ['상태 미확인', 'neutral'];
      const item = node('div', null, 'model-item');
      const title = node('div');
      title.append(node('h3', label), node('p', model.id || '설정 미확인', 'model-id'));
      item.append(
        title,
        tag(statusLabel, tone),
        node(
          'p',
          `설정 버전 ${model.configured_revision || '미확인'} · 확인된 리비전 ${model.resolved_revision || '미확인'}${model.adapter ? ` · ${model.adapter}` : ''}`,
          'model-revision',
        ),
      );
      el('model-list').append(item);
    }
    facts('environment-facts', [
      ['콘텐츠 유형', sample ? '가상 샘플 데이터' : '공개 미디어 메타데이터'],
      ['인덱스 상태', indexState(data.index?.state)],
      ['메타데이터 적재', data.capabilities?.ingest_execute ? '실행 가능' : '검증 미리보기만 가능'],
      ['작업 처리', '요청 내 동기 처리'],
      ['추론 준비 확인', '상태 조회에서는 미검증'],
    ]);
    el('catalog-notice').textContent = data.notice || '';
    el('manifest-limit').textContent =
      `최대 ${formatNumber(data.limits?.max_batch_size || 100)}개 항목 · ${formatNumber((data.limits?.max_body_bytes || 2000000) / 1000000)} MB`;
    el('ingest-availability').textContent = data.capabilities?.ingest_execute
      ? '검증을 통과한 매니페스트만 적재할 수 있습니다.'
      : '현재 환경에서는 미리보기만 가능합니다. 실제 적재에는 OpenSearch가 필요합니다.';
    el('search-mode-label').textContent = mock ? '모의 실행 · 모델 미사용' : '실제 모델 실행';
    el('search-mode-label').className = `tag ${mock ? 'amber' : 'teal'}`;
    if (!el('search-query').value)
      el('search-query').placeholder = sample ? '예: 5살 아이가 볼 공룡 영상' : '예: 펭귄 영상';
    const errors = Array.isArray(data.errors) ? data.errors : [];
    if (errors.length)
      feedback(
        'global-feedback',
        errors.map((item) => `${item.code || '상태 확인'}: ${item.message}`).join('\n'),
        'warning',
      );
    else if (data.operation_in_progress)
      feedback(
        'global-feedback',
        '다른 관리 작업이 진행 중입니다. 작업 완료 후 새로고침해 주세요.',
        'warning',
      );
    else feedback('global-feedback');
    renderIndex(data.index || {});
    applyControls();
  }

  function indexState(value) {
    return (
      {
        not_applicable: '샘플 환경 · 해당 없음',
        missing: '생성 전',
        available: '존재 확인',
        unavailable: '연결 확인 필요',
      }[value] || '미확인'
    );
  }

  function renderIndex(index) {
    const healthLabels = {
      green: 'Green · 정상',
      yellow: 'Yellow · 복제본 상태 확인',
      red: 'Red · 장애 확인 필요',
    };
    el('index-state-label').textContent = indexState(index.state);
    const tone =
      index.state === 'unavailable' || index.health === 'red'
        ? 'red'
        : index.health === 'yellow'
          ? 'amber'
          : index.state === 'available'
            ? 'teal'
            : 'neutral';
    el('index-state-label').className = `tag ${tone}`;
    facts('index-facts', [
      ['인덱스 이름', index.name || '해당 없음'],
      ['인덱스 상태', indexState(index.state)],
      [
        '문서 수',
        Number.isFinite(index.document_count)
          ? `${formatNumber(index.document_count)}개`
          : '미확인',
      ],
      ['인덱스 헬스', healthLabels[index.health] || '미확인'],
      ['스키마', index.schema || '미확인'],
      ['임베딩 호환성', '상태 조회에서 미검증'],
    ]);
    el('index-notice').textContent =
      index.message || '인덱스 준비를 실행하면 현재 모델 구성과의 호환성을 확인합니다.';
    el('index-identity-details').hidden = !index.embedding_identity;
    el('index-identity').textContent = index.embedding_identity
      ? JSON.stringify(index.embedding_identity, null, 2)
      : '';
    el('index-availability').textContent =
      state.status?.backend === 'opensearch'
        ? '작업 대상은 서버에 설정된 현재 인덱스입니다.'
        : '샘플 환경에서는 인덱스 관리 작업이 비활성화됩니다. OpenSearch 백엔드에서 사용할 수 있습니다.';
  }

  async function refreshStatus() {
    if (!state.connected) return;
    el('refresh-status').disabled = true;
    try {
      renderStatus(await api('/status'));
    } catch (error) {
      if (error.name !== 'AbortError') feedback('global-feedback', errorMessage(error), 'error');
    } finally {
      applyControls();
    }
  }

  function activateView(view, focus = false) {
    if (!state.connected || !views[view]) return;
    state.view = view;
    document.querySelectorAll('[data-view]').forEach((button) => {
      const active = button.dataset.view === view;
      button.classList.toggle('active', active);
      button.setAttribute('aria-selected', String(active));
      button.tabIndex = active ? 0 : -1;
      el(`panel-${button.dataset.view}`).hidden = !active;
    });
    el('page-title').textContent = views[view][0];
    el('breadcrumb-view').textContent = views[view][0];
    el('page-description').textContent = views[view][1];
    if (focus) el(`tab-${view}`).focus();
    if (view === 'contents' && !state.contentLoaded && !state.contentLoading) loadContents(0);
  }

  function disconnect(message = '') {
    state.session += 1;
    for (const controller of state.controllers) controller.abort();
    state.controllers.clear();
    state.key = '';
    state.connected = false;
    state.status = null;
    state.preview = null;
    state.fileRequest += 1;
    state.searchVerification = null;
    state.contentLoaded = false;
    state.contentOffset = 0;
    state.contentTotal = 0;
    state.contentQuery = '';
    state.contentPageSize = 20;
    state.contentHasMore = false;
    state.contentRequest += 1;
    state.contentLoading = false;
    state.searchPending = false;
    state.previewPending = false;
    state.writePending = false;
    state.confirmAction = null;
    el('connection-form').reset();
    el('connection-gate').hidden = false;
    el('workspace').hidden = true;
    el('refresh-status').hidden = true;
    el('disconnect').hidden = true;
    el('connection-state').classList.remove('connected');
    el('connection-label').textContent = '연결 전';
    el('sidebar-environment').textContent = '연결 대기';
    el('sidebar-note').textContent = '관리자 키로 연결하면 운영 상태를 확인할 수 있습니다.';
    for (const id of [
      'content-rows',
      'model-list',
      'environment-facts',
      'index-facts',
      'search-results',
      'search-summary',
      'search-conditions',
      'search-notice',
      'search-raw',
      'ingest-result',
      'ingest-raw',
      'index-identity',
      'index-operation-raw',
      'content-detail-title',
      'content-detail-description',
      'content-detail-facts',
      'content-detail-source',
      'content-detail-raw',
      'confirm-description',
      'confirm-note',
      'catalog-notice',
    ])
      el(id).replaceChildren();
    for (const id of [
      'stat-content',
      'stat-backend',
      'stat-mode',
      'stat-model',
      'stat-content-note',
      'stat-backend-note',
      'stat-mode-note',
      'stat-model-note',
      'content-count-label',
      'index-state-label',
      'index-notice',
      'overview-status',
      'overview-summary',
      'last-checked',
    ])
      el(id).textContent = '확인 전';
    for (const id of ['ingest-form', 'search-form', 'content-filter']) el(id).reset();
    for (const id of [
      'global-feedback',
      'content-feedback',
      'ingest-feedback',
      'search-feedback',
      'index-feedback',
    ])
      feedback(id);
    for (const id of [
      'search-output',
      'ingest-result',
      'ingest-raw-details',
      'index-operation-details',
      'index-identity-details',
    ])
      el(id).hidden = true;
    el('search-initial').hidden = false;
    document.querySelectorAll('[data-view]').forEach((button) => {
      button.disabled = true;
    });
    for (const id of ['content-dialog', 'confirm-dialog']) if (el(id).open) el(id).close();
    feedback('connection-feedback', message, message ? 'error' : '');
    applyControls();
    el('admin-key').focus();
  }

  el('connection-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    if (state.connected || el('connect-button').disabled) return;
    state.key = el('admin-key').value;
    el('admin-key').value = '';
    el('connect-button').disabled = true;
    el('connect-button').textContent = '연결 확인 중…';
    feedback('connection-feedback', '관리자 권한과 서비스 상태를 확인하고 있습니다.');
    try {
      const data = await api('/status');
      state.connected = true;
      el('workspace').hidden = false;
      el('connection-gate').hidden = true;
      el('refresh-status').hidden = false;
      el('disconnect').hidden = false;
      el('connection-state').classList.add('connected');
      el('connection-label').textContent = '관리자 연결됨';
      document.querySelectorAll('[data-view]').forEach((button) => {
        button.disabled = false;
      });
      if (data.defaults) {
        el('search-top-k').value = data.defaults.top_k;
        el('search-top-n').value = data.defaults.top_n;
      }
      renderStatus(data);
      activateView('overview', true);
      feedback('connection-feedback');
    } catch (error) {
      state.key = '';
      if (error.name !== 'AbortError')
        feedback('connection-feedback', errorMessage(error), 'error');
      el('admin-key').focus();
    } finally {
      el('connect-button').disabled = false;
      el('connect-button').textContent = '운영 콘솔 연결 →';
    }
  });
  el('disconnect').addEventListener('click', () => {
    if (!state.writePending) disconnect();
  });
  el('refresh-status').addEventListener('click', refreshStatus);
  document.querySelectorAll('[data-view]').forEach((button) => {
    button.addEventListener('click', () => activateView(button.dataset.view));
    button.addEventListener('keydown', (event) => {
      const names = Object.keys(views);
      let index = names.indexOf(button.dataset.view);
      if (['ArrowDown', 'ArrowRight'].includes(event.key)) index = (index + 1) % names.length;
      else if (['ArrowUp', 'ArrowLeft'].includes(event.key))
        index = (index + names.length - 1) % names.length;
      else if (event.key === 'Home') index = 0;
      else if (event.key === 'End') index = names.length - 1;
      else return;
      event.preventDefault();
      activateView(names[index], true);
    });
  });
  document
    .querySelectorAll('[data-open-view]')
    .forEach((button) =>
      button.addEventListener('click', () => activateView(button.dataset.openView, true)),
    );

  async function loadContents(offset = 0, useCurrentFilters = true) {
    if (!state.connected) return;
    const request = ++state.contentRequest;
    state.contentLoading = true;
    feedback('content-feedback', '등록 콘텐츠를 조회하고 있습니다.');
    applyControls();
    const query = useCurrentFilters ? el('content-query').value.trim() : state.contentQuery;
    const limit = useCurrentFilters ? Number(el('content-limit').value) : state.contentPageSize;
    const params = new URLSearchParams({ q: query, offset: String(offset), limit: String(limit) });
    try {
      const data = await api(`/contents?${params}`);
      if (request !== state.contentRequest) return;
      state.contentLoaded = true;
      state.contentOffset = data.offset;
      state.contentTotal = data.total;
      state.contentQuery = query;
      state.contentPageSize = data.limit;
      state.contentHasMore =
        Boolean(data.has_more) &&
        data.offset + data.limit < (state.status?.limits?.max_result_window || 10000);
      const items = Array.isArray(data.items) ? data.items : [];
      el('content-rows').replaceChildren();
      for (const item of items) {
        const row = node('tr');
        const main = node('td');
        const contentId = node('span', item.content_id, 'content-id');
        contentId.title = item.content_id;
        main.append(
          node('span', item.title, 'content-title'),
          node('span', item.description, 'content-description'),
          contentId,
        );
        const topics = node('td');
        const wrap = node('div', null, 'tags-wrap');
        for (const topic of (item.tags || []).slice(0, 3)) wrap.append(tag(topic));
        if ((item.tags || []).length > 3) wrap.append(tag(`+${item.tags.length - 3}`));
        topics.append(wrap.childElementCount ? wrap : node('span', '—'));
        const source = node(
          'td',
          item.is_sample ? '가상 샘플' : `${item.license || item.language || '미확인'}${item.rights_scope === 'metadata' ? ' (메타데이터만)' : ''}`,
          'table-source',
        );
        const action = node('td');
        const button = node('button', '보기', 'button button-quiet small');
        button.type = 'button';
        button.setAttribute('aria-label', `${item.title} 상세 보기`);
        button.addEventListener('click', () => showContent(item));
        action.append(button);
        row.append(main, topics, node('td', ageLabel(item)), source, action);
        el('content-rows').append(row);
      }
      el('content-empty').hidden = items.length > 0;
      el('content-count-label').textContent = `검색 결과 ${formatNumber(data.total)}개`;
      el('content-page-label').textContent = items.length
        ? `${formatNumber(data.offset + 1)}–${formatNumber(data.offset + items.length)} / ${formatNumber(data.total)}개`
        : `0 / ${formatNumber(data.total)}개`;
      feedback('content-feedback');
      if (data.window_limited)
        feedback(
          'content-feedback',
          '목록은 처음 10,000건까지 조회할 수 있습니다. 검색어로 결과를 좁혀 주세요.',
          'warning',
        );
    } catch (error) {
      if (request === state.contentRequest && error.name !== 'AbortError') {
        feedback('content-feedback', errorMessage(error), 'error');
        el('content-rows').replaceChildren();
        el('content-count-label').textContent = '조회 실패';
        el('content-page-label').textContent = '조회 결과를 확인할 수 없습니다.';
        state.contentHasMore = false;
        el('content-empty').hidden = true;
      }
    } finally {
      if (request === state.contentRequest) {
        state.contentLoading = false;
        applyControls();
      }
    }
  }
  el('content-filter').addEventListener('submit', (event) => {
    event.preventDefault();
    loadContents(0);
  });
  el('content-limit').addEventListener('change', () => loadContents(0));
  el('content-prev').addEventListener('click', () =>
    loadContents(Math.max(0, state.contentOffset - state.contentPageSize), false),
  );
  el('content-next').addEventListener('click', () =>
    loadContents(state.contentOffset + state.contentPageSize, false),
  );

  function showContent(item) {
    el('content-detail-title').textContent = item.title;
    el('content-detail-description').textContent = item.description;
    facts('content-detail-facts', [
      ['콘텐츠 ID', item.content_id],
      ['주제', listText(item.tags)],
      ['캐릭터', listText(item.characters)],
      ['연령', ageLabel(item)],
      [
        '길이',
        Number.isFinite(item.duration_seconds)
          ? `${formatNumber(item.duration_seconds)}초`
          : '미상',
      ],
      ['언어', item.language || '미상'],
      ['라이선스', item.is_sample ? '가상 샘플' : `${item.license || '미확인'}${item.rights_scope === 'metadata' ? ' (메타데이터만; 영상 권리 미확인)' : ''}`],
      ['원본 ID', item.source_id || '해당 없음'],
    ]);
    el('content-detail-source').replaceChildren();
    const source = safeLink(item.canonical_url, '원본 출처 페이지 열기 ↗');
    if (source) el('content-detail-source').append(source);
    el('content-detail-raw').textContent = JSON.stringify(item, null, 2);
    el('content-dialog').showModal();
  }
  el('close-content-dialog').addEventListener('click', () => el('content-dialog').close());

  el('search-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    if (el('run-search').disabled) return;
    const query = el('search-query').value.trim();
    const topK = Number(el('search-top-k').value);
    const topN = Number(el('search-top-n').value);
    if (!query) {
      feedback('search-feedback', '검색어를 입력해 주세요.', 'error');
      return;
    }
    if (topN > topK) {
      feedback('search-feedback', '결과 Top N은 후보 Top K 이하여야 합니다.', 'error');
      return;
    }
    state.searchPending = true;
    el('search-output').hidden = true;
    el('search-initial').hidden = true;
    el('search-results').replaceChildren();
    el('search-raw').textContent = '';
    feedback(
      'search-feedback',
      state.status?.mode === 'mock'
        ? '모의 검색을 실행하고 있습니다.'
        : '검색을 실행하고 있습니다. 첫 모델 로딩은 오래 걸릴 수 있습니다.',
    );
    applyControls();
    try {
      const data = await api('/search', { query, top_k: topK, top_n: topN });
      renderSearch(data);
      if (data.model_inference_performed) state.searchVerification = nowLabel();
      feedback(
        'search-feedback',
        `${data.results.length}개 결과 · 후보 ${formatNumber(data.candidate_count)}개 · ${formatNumber(data.elapsed_ms)} ms`,
        'success',
      );
      await refreshStatus();
    } catch (error) {
      if (error.name !== 'AbortError')
        feedback('search-feedback', `검색 실패: ${errorMessage(error)}`, 'error');
      el('search-initial').hidden = false;
    } finally {
      state.searchPending = false;
      applyControls();
    }
  });

  function renderSearch(data) {
    el('search-output').hidden = false;
    el('search-raw').textContent = JSON.stringify(data, null, 2);
    el('search-summary').replaceChildren();
    for (const [label, value] of [
      ['검색 결과', `${data.results.length}개`],
      ['후보 콘텐츠', `${formatNumber(data.candidate_count)}개`],
      ['응답 시간', `${formatNumber(data.elapsed_ms)} ms`],
    ]) {
      const chip = node('span', label, 'summary-chip');
      chip.append(node('strong', value));
      el('search-summary').append(chip);
    }
    el('search-summary').append(
      node(
        'span',
        data.model_inference_performed ? '실제 모델 추론 수행' : '모델 추론 미실행',
        `summary-chip${data.model_inference_performed ? ' verified' : ''}`,
      ),
    );
    const conditions = data.conditions || {};
    el('search-conditions').replaceChildren(
      tag(
        `연령 ${conditions.age === null || conditions.age === undefined ? '미지정' : `${conditions.age}세`}`,
      ),
      tag(`주제 ${listText(conditions.topics)}`),
      tag(`캐릭터 ${listText(conditions.characters)}`),
    );
    el('search-notice').textContent = [data.sample_notice, data.notice, '점수는 확률이 아닙니다.']
      .filter(Boolean)
      .join(' ');
    if (!data.results.length) {
      const empty = node('section', null, 'card empty-state');
      empty.append(
        node('h3', '조건에 맞는 결과가 없습니다'),
        node(
          'p',
          '검색어와 연령 조건을 확인해 주세요. 연령 미상 자료는 나이를 지정한 검색에서 제외됩니다.',
        ),
      );
      el('search-results').append(empty);
    }
    data.results.forEach((result, index) => {
      const content = result.content || {};
      const card = node('article', null, 'card result-card');
      card.append(
        node('span', String(index + 1).padStart(2, '0'), 'result-rank'),
        node('h3', content.title),
        node('p', content.description),
      );
      if (Array.isArray(result.evidence) && result.evidence.length) {
        const evidence = node('ul', null, 'result-evidence');
        result.evidence.forEach((item) => evidence.append(node('li', item)));
        card.append(evidence);
      }
      const meta = node('div', null, 'result-meta');
      meta.append(tag(`연령 ${ageLabel(content)}`));
      (result.retrieval_sources || []).forEach((source) => meta.append(tag(source, 'teal')));
      if (Number.isFinite(result.reranker_score))
        meta.append(tag(`BGE ${result.reranker_score.toFixed(4)}`));
      if (Number.isFinite(result.fusion_score))
        meta.append(tag(`RRF ${result.fusion_score.toFixed(4)}`));
      card.append(meta);
      const id = node('span', result.content_id || content.id || '', 'content-id');
      id.title = id.textContent;
      card.append(id);
      const source = safeLink(content.canonical_url, '출처·라이선스 페이지 ↗');
      if (source) card.append(source);
      el('search-results').append(card);
    });
  }

  function invalidatePreview() {
    state.preview = null;
    el('ingest-result').hidden = true;
    el('ingest-result').replaceChildren();
    el('ingest-raw-details').hidden = true;
    el('ingest-raw').textContent = '';
    feedback('ingest-feedback');
    applyControls();
  }
  el('manifest-json').addEventListener('input', () => {
    state.fileRequest += 1;
    invalidatePreview();
  });
  el('manifest-file').addEventListener('change', async () => {
    const fileRequest = ++state.fileRequest;
    invalidatePreview();
    const file = el('manifest-file').files[0];
    if (!file) return;
    el('manifest-json').value = '';
    const session = state.session;
    const maxBytes = state.status?.limits?.max_body_bytes || 2000000;
    if (file.size > maxBytes) {
      feedback(
        'ingest-feedback',
        `파일은 ${formatNumber(maxBytes / 1000000)} MB 이하여야 합니다.`,
        'error',
      );
      el('manifest-file').value = '';
      return;
    }
    try {
      const text = await file.text();
      if (session === state.session && fileRequest === state.fileRequest) {
        el('manifest-json').value = text;
        invalidatePreview();
      }
    } catch {
      feedback(
        'ingest-feedback',
        '파일을 읽지 못했습니다. JSON 내용을 직접 붙여 넣을 수 있습니다.',
        'error',
      );
    }
  });

  el('ingest-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    if (el('preview-ingest').disabled) return;
    invalidatePreview();
    let manifest;
    try {
      manifest = JSON.parse(el('manifest-json').value);
      const maxBytes = state.status?.limits?.max_body_bytes || 2000000;
      if (
        new TextEncoder().encode(JSON.stringify({ manifest, dry_run: false, confirmed: true }))
          .length > maxBytes
      )
        throw new Error(`요청은 ${formatNumber(maxBytes / 1000000)} MB 이하여야 합니다.`);
      if (!manifest || !Array.isArray(manifest.entries) || !manifest.entries.length)
        throw new Error('매니페스트에 entries 배열과 콘텐츠 항목이 필요합니다.');
      if (manifest.entries.length > (state.status?.limits?.max_batch_size || 100))
        throw new Error(
          `한 번에 최대 ${state.status?.limits?.max_batch_size || 100}개 항목을 검증할 수 있습니다.`,
        );
    } catch (error) {
      feedback(
        'ingest-feedback',
        error instanceof SyntaxError
          ? 'JSON 형식이 올바르지 않습니다. 괄호와 쉼표를 확인해 주세요.'
          : error.message,
        'error',
      );
      return;
    }
    state.previewPending = true;
    applyControls();
    feedback('ingest-feedback', '메타데이터 형식과 적재 계획을 검증하고 있습니다.');
    try {
      const data = await api('/ingest', { manifest, dry_run: true });
      state.preview = { manifest, response: data };
      renderIngest(data);
      feedback(
        'ingest-feedback',
        `${data.entries.length}개 항목 검증 완료. 아직 인덱스에 적재하지 않았습니다.`,
        'success',
      );
    } catch (error) {
      if (error.name !== 'AbortError') feedback('ingest-feedback', errorMessage(error), 'error');
    } finally {
      state.previewPending = false;
      applyControls();
    }
  });

  function renderIngest(data) {
    const target = el('ingest-result');
    target.hidden = false;
    target.replaceChildren();
    target.append(
      node(
        'h3',
        data.dry_run
          ? `검증된 콘텐츠 ${data.entries.length}개`
          : `적재 ${formatNumber(data.indexed)}개 / 전체 ${data.entries.length}개`,
      ),
    );
    const list = node('ul', null, 'receipt-list');
    for (const item of data.entries) {
      const row = node('li');
      row.append(
        tag(
          data.dry_run ? '검증됨' : item.status === 'indexed' ? '적재됨' : '실패',
          data.dry_run || item.status === 'indexed' ? 'teal' : 'red',
        ),
      );
      const description = node('div');
      description.append(node('p', item.title || item.content_id));
      description.append(
        node(
          'small',
          data.dry_run
            ? `${item.license || '라이선스 미확인'} · 연령 ${item.age_known ? '확인됨' : '미상'}`
            : [item.stage, item.error_type].filter(Boolean).join(' · '),
        ),
      );
      row.append(description);
      list.append(row);
    }
    target.append(list);
    el('ingest-raw-details').hidden = false;
    el('ingest-raw').textContent = JSON.stringify(data, null, 2);
  }

  function confirmOperation(title, description, note, button, action) {
    if (state.writePending) return;
    el('confirm-title').textContent = title;
    el('confirm-description').textContent = description;
    el('confirm-note').textContent = note;
    el('accept-confirm').textContent = button;
    state.confirmAction = action;
    el('confirm-dialog').showModal();
    el('cancel-confirm').focus();
  }
  el('cancel-confirm').addEventListener('click', () => {
    state.confirmAction = null;
    el('confirm-dialog').close();
  });
  el('confirm-dialog').addEventListener('cancel', () => {
    state.confirmAction = null;
  });
  el('accept-confirm').addEventListener('click', async () => {
    const action = state.confirmAction;
    state.confirmAction = null;
    el('confirm-dialog').close();
    if (action && state.connected && !state.writePending) await action();
  });

  el('execute-ingest').addEventListener('click', () => {
    if (!state.preview || el('execute-ingest').disabled) return;
    const preview = state.preview;
    confirmOperation(
      '콘텐츠를 인덱스에 적재할까요?',
      `${preview.response.entries.length}개 콘텐츠 → ${state.status?.index?.name || '설정된 인덱스'}\n같은 콘텐츠 ID가 있으면 최신 메타데이터로 갱신됩니다.`,
      '실제 임베딩 모델을 실행합니다. 첫 로딩은 오래 걸릴 수 있으며, 처리 중에는 작업을 다시 요청할 수 없습니다.',
      '적재 실행',
      async () => {
        state.writePending = true;
        state.preview = null;
        applyControls();
        feedback(
          'ingest-feedback',
          '메타데이터를 적재하고 있습니다. 모델 로딩과 임베딩 생성에 시간이 걸릴 수 있습니다. 이 창에서 결과를 기다려 주세요.',
        );
        try {
          const data = await api('/ingest', {
            manifest: preview.manifest,
            dry_run: false,
            confirmed: true,
          });
          renderIngest(data);
          feedback(
            'ingest-feedback',
            data.status === 'partial_failure'
              ? `${formatNumber(data.indexed)}개 적재 완료. 일부 항목이 실패했습니다. 아래 항목별 결과와 서버 기록을 확인해 주세요.`
              : `${formatNumber(data.indexed)}개 적재 완료. 인덱스 새로고침 후 콘텐츠 목록에서 확인하세요.`,
            data.status === 'partial_failure' ? 'warning' : 'success',
          );
          state.contentLoaded = false;
          await refreshStatus();
        } catch (error) {
          if (error.name !== 'AbortError')
            feedback('ingest-feedback', errorMessage(error, true), 'error');
        } finally {
          state.writePending = false;
          applyControls();
        }
      },
    );
  });

  async function indexOperation(path, pendingText) {
    state.writePending = true;
    applyControls();
    feedback('index-feedback', pendingText);
    el('index-operation-details').hidden = true;
    el('index-operation-raw').textContent = '';
    try {
      const data = await api(path, { confirmed: true });
      el('index-operation-details').hidden = false;
      el('index-operation-raw').textContent = JSON.stringify(data, null, 2);
      feedback(
        'index-feedback',
        data.status === 'refreshed'
          ? `${data.index} 인덱스를 새로 고쳤습니다.`
          : `${data.index} 인덱스 준비가 완료되었습니다. 실제 검색 추론은 검색 테스트에서 확인하세요.`,
        'success',
      );
      state.contentLoaded = false;
      await refreshStatus();
    } catch (error) {
      if (error.name !== 'AbortError')
        feedback('index-feedback', errorMessage(error, true), 'error');
    } finally {
      state.writePending = false;
      applyControls();
    }
  }
  el('ensure-index').addEventListener('click', () => {
    if (el('ensure-index').disabled) return;
    confirmOperation(
      '검색 인덱스를 준비할까요?',
      `대상 인덱스: ${state.status?.index?.name}\n인덱스가 없으면 생성하고, 기존 인덱스는 현재 임베딩 구성과의 호환성을 확인합니다.`,
      '임베딩 모델 로딩이 발생할 수 있습니다. 인덱스 준비 완료는 실제 검색 추론 성공을 의미하지 않습니다.',
      '준비 실행',
      () =>
        indexOperation(
          '/index/ensure',
          '인덱스를 준비하고 있습니다. 임베딩 모델 로딩은 오래 걸릴 수 있습니다.',
        ),
    );
  });
  el('refresh-index').addEventListener('click', () => {
    if (el('refresh-index').disabled) return;
    confirmOperation(
      '최근 문서를 검색에 반영할까요?',
      `대상 인덱스: ${state.status?.index?.name}`,
      '현재 인덱스를 새로 고쳐 최근 적재한 문서를 검색할 수 있도록 반영합니다.',
      '새로고침 실행',
      () => indexOperation('/index/refresh', '인덱스를 새로 고치고 있습니다.'),
    );
  });

  // Never retain credentials in history, cookies, or browser storage.
  window.addEventListener('pagehide', () => {
    state.key = '';
    state.preview = null;
    for (const controller of state.controllers) controller.abort();
    el('admin-key').value = '';
  });
  window.addEventListener('pageshow', (event) => {
    if (event.persisted) disconnect();
  });
})();
