// src/pages/MachineLearning/CreateDatasetPage.jsx
import React, { useEffect, useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Container,
  Grid,
  Paper,
  Stack,
  Box,
  Typography,
  TextField,
  Button,
  CircularProgress,
  Autocomplete,
  Chip,
  Divider,
  Card,
  CardActionArea,
  CardHeader,
  CardContent,
  Avatar,
  Tooltip,
  Drawer,
  Dialog,
  List,
  ListItem,
  ListItemText,
  useTheme,
  Fab,
  useMediaQuery,
  Checkbox,
  ButtonGroup,
  IconButton,
} from '@mui/material';

import {
  Close as CloseIcon,
  Visibility as VisibilityIcon,
  FilterList as FilterListIcon,
  SelectAll as SelectAllIcon,
  NoteAdd as NoteAddIcon,
  HelpOutline as HelpOutlineIcon,
  Biotech as BacteriaIcon,
  Label as SampleIcon,
  Comment as CommentIcon,
  Flag as FlagIcon,
  Schedule as ScheduleIcon,
} from '@mui/icons-material';
import dayjs from 'dayjs';
import 'dayjs/locale/pt-br';

//manter o padrão no service
import { getExperimentos } from '../../infra/api/ExperimentoApi';
import { getProtocolos } from '../../infra/api/Protocols';
import { getTowers } from '../../infra/api/Torres';
import { getBacterias } from '../../infra/api/Bacteria'; 
import QuickGraficoViewer from '../../components/QuickGraficoViewer';

dayjs.locale('pt-br');

async function fetchAllTowers() {
  const first = await getTowers(0);
  let all = first.content || [];
  for (let i = 1; i < first.totalPages; i++) {
    const page = await getTowers(i);
    all = all.concat(page.content || []);
  }
  return all;
}

const labelData = dateString => {
  const d = dayjs(dateString);
  const diff = dayjs().startOf('day').diff(d.startOf('day'), 'day');
  if (diff === 0) return 'Hoje';
  if (diff === 1) return 'Ontem';
  if (diff < 7) return d.format('dddd').replace(/^./, s => s.toUpperCase());
  return d.format('DD/MM/YYYY');
};

export default function CreateDatasetPage() {
  const theme = useTheme();
  const navigate = useNavigate();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  // Data states
  const [experiments, setExperiments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Bactérias para filtro de calibração
  const [bacterias, setBacterias] = useState([]);

  // Estados temporários de filtros
  const [fCli, setFCli] = useState([]);
  const [fCid, setFCid] = useState([]);
  const [fProt, setFProt] = useState(null);
  const [fAna, setFAna] = useState([]);
  const [fCal, setFCal] = useState([]); // novo

  // Filtros aplicados
  const [aCli, setACli] = useState([]);
  const [aCid, setACid] = useState([]);
  const [aProt, setAProt] = useState(null);
  const [aAna, setAAna] = useState([]);
  const [aCal, setACal] = useState([]); // novo

  // Seleção + dataset
  const [selIds, setSelIds] = useState(new Set());
  const [dsName, setDsName] = useState('');
  const [openSummary, setOpenSummary] = useState(false);

  // Drawer de preview
  const [previewExp, setPreviewExp] = useState(null);

  // 1) Buscar experimentos + protocolos + torres
  useEffect(() => {
    (async () => {
      setLoading(true);
      try {
        const [exps, prots] = await Promise.all([
          getExperimentos(),
          getProtocolos(),
        ]);
        const towers = await fetchAllTowers();
        const mapT = new Map(towers.map(t => [t.numeroSerie, t]));

        const mapped = exps
          .filter(e => e.general_info.status === 'completed')
          .map(e => {
            const proto = prots.find(p => p.id === e.general_info.protocolId) || {};
            const ana = proto.analysis?.find(a => a.id === e.general_info.analysisId) || {};
            const tow = mapT.get(e.numeroSerie) || {};
            return {
              ...e,
              serialNumber: e.numeroSerie,
              protocolName: proto.name || '—',
              analysisName: ana.name || '—',
              deviceName: tow.deviceName || '—',
              client: tow.idCliente,
              city: tow.endereco?.cidade,
              // Garantir que exista calibrationSets
              calibrationSets: Array.isArray(proto.calibrationSets)
                ? proto.calibrationSets
                : [],
            };
          });
        setExperiments(mapped);
      } catch {
        setError('Erro ao carregar dados.');
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  // 2) Buscar lista de bactérias
  useEffect(() => {
    getBacterias()
      .then(data => Array.isArray(data) && setBacterias(data))
      .catch(() => setBacterias([]));
  }, []);

  // 3) Map de id→nome
  const bactMap = useMemo(
    () => Object.fromEntries(bacterias.map(b => [b.id, b.name])),
    [bacterias]
  );

  // Opções derivadas de filtros existentes
  const optCli = useMemo(
    () => [...new Set(experiments.map(e => e.client).filter(Boolean))],
    [experiments]
  );
  const optCid = useMemo(
    () => [...new Set(experiments.map(e => e.city).filter(Boolean))],
    [experiments]
  );
  const optProt = useMemo(
    () => [...new Set(experiments.map(e => e.protocolName).filter(Boolean))],
    [experiments]
  );
  const optAna = useMemo(
    () => [
      ...new Set(
        experiments
          .filter(e => !fProt || e.protocolName === fProt)
          .map(e => e.analysisName)
          .filter(Boolean)
      ),
    ],
    [experiments, fProt]
  );
  const optCal = useMemo(() => {
    const labels = experiments
      .filter(e => !aProt || e.protocolName === aProt)
      .flatMap(e => e.calibrationSets)
      .map(c => {
        const name = bactMap[c.bacteriaId] || '–';
        return `${name} (${c.unit})`;
      });
    return [...new Set(labels)];
  }, [experiments, aProt, bactMap]);


  // 4) Aplicar todos os filtros, incluindo calibrações
  const filtered = useMemo(
    () =>
      experiments
        .filter(
          e =>
            (!aCli.length || aCli.includes(e.client)) &&
            (!aCid.length || aCid.includes(e.city)) &&
            (!aProt || e.protocolName === aProt) &&
            (!aAna.length || aAna.includes(e.analysisName)) &&
            (!aCal.length ||
              e.calibrationSets.some(c => {
                const label = `${bactMap[c.bacteriaId]} (${c.unit})`;
                return aCal.includes(label);
              })
            )

        ),
    [experiments, aCli, aCid, aProt, aAna, aCal, bactMap]
  );

  // Deselecionar IDs que ficaram ocultos
  useEffect(() => {
    setSelIds(prev =>
      new Set([...prev].filter(id => filtered.some(e => e.id === id)))
    );
  }, [filtered]);

  // Agrupar por data
  const groups = useMemo(() => {
    const byDate = {};
    filtered.forEach(e => {
      const d = dayjs.unix(+e.general_info.start_date).format('YYYY-MM-DD');
      (byDate[d] = byDate[d] || []).push(e);
    });
    return Object.entries(byDate)
      .map(([date, items]) => ({ date, label: labelData(date), items }))
      .sort((a, b) => (a.date < b.date ? 1 : -1));
  }, [filtered]);

  // Métricas
  const total = experiments.length;
  const hasApplied =
    aCli.length ||
    aCid.length ||
    aProt ||
    aAna.length ||
    aCal.length;

  // Handlers de seleção
  const toggle = id =>
    setSelIds(s => {
      const n = new Set(s);
      n.has(id) ? n.delete(id) : n.add(id);
      return n;
    });
  const toggleAll = () =>
    setSelIds(new Set(filtered.map(e => e.id)));
  const clearSelection = () => setSelIds(new Set());



  // Ao confirmar, loga o JSON com nome, filtros e experiments
  const handleConfirm = () => {
    const payload = {
      datasetName: dsName,
      filters: {
        clients: aCli,
        cities: aCid,
        protocol: aProt,
        analyses: aAna,
        calibrations: aCal,
      },
      experiments: [...selIds],
    };
    console.log(JSON.stringify(payload, null, 2));
    navigate('/machine-learning/my-datasets');
  };


  if (loading) {
    return (
      <Container sx={{ textAlign: 'center', py: 6 }}>
        <CircularProgress />
        <Typography mt={2}>Carregando…</Typography>
      </Container>
    );
  }
  if (error) {
    return <Typography color="error">{error}</Typography>;
  }

  return (
    <Container maxWidth="xl" sx={{ pb: 4 }}>
      {/* Header */}
      <Stack direction="row" spacing={2} alignItems="center" mb={3}>
        <Button variant="outlined" onClick={() => navigate(-1)}>
          Voltar
        </Button>
        <Typography variant="h4">Criar Dataset</Typography>
      </Stack>

      <Grid container spacing={2}>
        {/* Centro: cartões */}
        <Grid item xs={12} md={8} lg={9} sx={{ order: { xs: 3, md: 1 } }}>
          {!hasApplied ? (
            <Typography
              color="text.secondary"
              align="center"
              mt={4}
            >
              Aplique ao menos um filtro para visualizar ensaios.
            </Typography>
          ) : (
            groups.map(({ date, label, items }) => {
              const allSel = items.every(e => selIds.has(e.id));
              return (
                <Box key={date} mb={4}>
                  <Box
                    display="flex"
                    justifyContent="space-between"
                    mb={1}
                  >
                    <Typography variant="h6">{label}</Typography>
                    <Checkbox
                      checked={allSel}
                      onChange={() => {
                        const n = new Set(selIds);
                        items.forEach(e =>
                          allSel ? n.delete(e.id) : n.add(e.id)
                        );
                        setSelIds(n);
                      }}
                    />
                  </Box>
                  <Grid container spacing={2}>
                    {items.map(exp => (
                      <Grid item xs={12} sm={6} md={4} key={exp.id}>
                        <Tooltip title="Visualizar gráficos">
                          <Card
                            variant="outlined"
                            sx={{
                              border: selIds.has(exp.id)
                                ? `2px solid ${theme.palette.primary.main}`
                                : undefined,
                            }}
                          >
                            <CardActionArea
                              onClick={() => setPreviewExp(exp)}
                            >
                              {/* ícone hover */}
                              <Box
                                sx={{
                                  position: 'absolute',
                                  top: 8,
                                  right: 8,
                                  p: 0.5,
                                  bgcolor: 'rgba(0,0,0,0.6)',
                                  borderRadius: 1,
                                  opacity: 0,
                                  '&:hover': { opacity: 1 },
                                }}
                              >
                                <VisibilityIcon
                                  fontSize="small"
                                  htmlColor="#fff"
                                />
                              </Box>
                              {/* Header */}
                              <CardHeader
                                avatar={
                                  <Avatar>
                                    {exp.protocolName[0] ||
                                      '—'}
                                  </Avatar>
                                }
                                action={
                                  <Checkbox
                                    checked={selIds.has(exp.id)}
                                    onChange={() => toggle(exp.id)}
                                    onClick={e =>
                                      e.stopPropagation()
                                    }
                                  />
                                }
                                title={exp.protocolName}
                                subheader={`${exp.analysisName} · ${exp.deviceName} (SN: ${exp.serialNumber})`}
                              />
                              {/* Cliente / Cidade */}
                              <Box
                                sx={{
                                  px: 2,
                                  pt: 0,
                                  pb: 1,
                                  display: 'flex',
                                  gap: 0.5,
                                  flexWrap: 'wrap',
                                }}
                              >
                                <Chip
                                  label={exp.client || '—'}
                                  size="small"
                                />
                                <Chip
                                  label={exp.city || '—'}
                                  size="small"
                                />
                              </Box>
                              {/* Conteúdo detalhado */}
                              <CardContent>
                                <Stack spacing={1}>
                                  {/* Datas */}
                                  <Stack
                                    direction="row"
                                    spacing={1}
                                    alignItems="center"
                                  >
                                    <ScheduleIcon fontSize="small" />
                                    <Typography variant="body2">
                                      Início:{' '}
                                      {dayjs
                                        .unix(
                                          +exp.general_info
                                            .start_date
                                        )
                                        .format(
                                          'DD/MM/YYYY HH:mm'
                                        )}
                                    </Typography>
                                  </Stack>
                                  {exp.general_info
                                    .end_date && (
                                      <Stack
                                        direction="row"
                                        spacing={1}
                                        alignItems="center"
                                      >
                                        <ScheduleIcon
                                          fontSize="small"
                                          color="success"
                                        />
                                        <Typography
                                          variant="body2"
                                          color="success.main"
                                        >
                                          Fim:{' '}
                                          {dayjs
                                            .unix(
                                              +
                                              exp
                                                .general_info
                                                .end_date
                                            )
                                            .format(
                                              'DD/MM/YYYY HH:mm'
                                            )}
                                        </Typography>
                                      </Stack>
                                    )}
                                  <Divider />
                                  {/* Bacteria Count */}
                                  <Stack
                                    direction="row"
                                    spacing={1}
                                    alignItems="center"
                                  >
                                    <BacteriaIcon fontSize="small" />
                                    <Typography variant="body2">
                                      {exp.general_info
                                        .bacteriaCount ??
                                        '–'}{' '}
                                      CFU
                                    </Typography>
                                  </Stack>
                                  <Divider />
                                  {/* Código da amostra */}
                                  <Stack
                                    direction="row"
                                    spacing={1}
                                    alignItems="center"
                                  >
                                    <SampleIcon fontSize="small" />
                                    <Typography variant="body2">
                                      {exp.codigo || '–'}
                                    </Typography>
                                  </Stack>
                                  <Divider />
                                  {/* Comentários */}
                                  <Stack spacing={0.5}>
                                    <Stack
                                      direction="row"
                                      spacing={1}
                                      alignItems="center"
                                    >
                                      <CommentIcon fontSize="small" />
                                      <Typography variant="subtitle2">
                                        Comentários
                                      </Typography>
                                    </Stack>
                                    {exp.comments?.length > 0 ? (
                                      exp.comments.map(
                                        (c, i) => (
                                          <Typography
                                            key={i}
                                            variant="body2"
                                            color="text.secondary"
                                          >
                                            • {c}
                                          </Typography>
                                        )
                                      )
                                    ) : (
                                      <Typography
                                        variant="body2"
                                        color="text.secondary"
                                      >
                                        Sem comentários
                                      </Typography>
                                    )}
                                  </Stack>
                                  <Divider />
                                  {/* Flag / Info coleta */}
                                  <Stack
                                    direction="row"
                                    spacing={1}
                                    alignItems="center"
                                  >
                                    <FlagIcon fontSize="small" />
                                    <Typography variant="body2">
                                      {exp.flagInfo ?? '–'}
                                    </Typography>
                                  </Stack>
                                </Stack>
                              </CardContent>
                            </CardActionArea>
                          </Card>
                        </Tooltip>
                      </Grid>
                    ))}
                  </Grid>
                </Box>
              );
            })
          )}
        </Grid>

        {/* Lateral direita: filtros, seleção e criar */}
        <Grid item xs={12} md={4} lg={3} sx={{ order: { xs: 2, md: 3 } }}>
          <Paper sx={{ p: 2, position: 'sticky', top: 72 }}>
            <Stack spacing={2}>
              {/* Filtrar */}
              <Box
                display="flex"
                justifyContent="space-between"
                alignItems="center"
              >
                <Stack direction="row" spacing={1} alignItems="center">
                  <FilterListIcon fontSize="small" />
                  <Typography variant="subtitle1">Filtrar</Typography>
                </Stack>
                <Tooltip title="Selecione um ou mais itens e a lista será atualizada automaticamente.">
                  <IconButton size="small">
                    <HelpOutlineIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
              <Stack spacing={1}>
                <Autocomplete
                  multiple
                  options={optCli}
                  value={fCli}
                  onChange={(_, v) => setFCli(v)}
                  renderInput={p => (
                    <TextField {...p} label="Clientes" size="small" />
                  )}
                />
                <Autocomplete
                  multiple
                  options={optCid}
                  value={fCid}
                  onChange={(_, v) => setFCid(v)}
                  renderInput={p => <TextField {...p} label="Cidades" size="small" />}
                />
                <Autocomplete
                  options={optProt}
                  value={fProt}
                  onChange={(_, v) => {
                    setFProt(v);
                    setFAna([]);
                    setFCal([]);
                  }}
                  renderInput={p => (
                    <TextField {...p} label="Protocolo" size="small" />
                  )}
                />
                <Autocomplete
                  multiple
                  options={optAna}
                  value={fAna}
                  onChange={(_, v) => setFAna(v)}
                  renderInput={p => (
                    <TextField {...p} label="Análises" size="small" />
                  )}
                  disabled={!fProt}
                />
                <Autocomplete
                  multiple
                  options={optCal}
                  value={fCal}
                  onChange={(_, v) => setFCal(v)}
                  renderInput={p => (
                    <TextField {...p} label="Calibrações" size="small" />
                  )}
                  disabled={!fProt}
                />
                <Stack direction="row" spacing={1}>
                  <Button
                    size="small"
                    variant="contained"
                    onClick={() => {
                      setACli(fCli);
                      setACid(fCid);
                      setAProt(fProt);
                      setAAna(fAna);
                      setACal(fCal); // novo
                    }}
                  >
                    Aplicar
                  </Button>
                  <Button
                    size="small"
                    onClick={() => {
                      setFCli([]);
                      setFCid([]);
                      setFProt(null);
                      setFAna([]);
                      setFCal([]); // novo
                      setACli([]);
                      setACid([]);
                      setAProt(null);
                      setAAna([]);
                      setACal([]); // novo
                    }}
                  >
                    Limpar
                  </Button>
                </Stack>
              </Stack>

              <Divider />

              {/* Seleção */}
              <Box>
                <Stack
                  direction="row"
                  justifyContent="space-between"
                  alignItems="center"
                >
                  <Stack direction="row" spacing={1} alignItems="center">
                    <SelectAllIcon fontSize="small" />
                    <Typography variant="subtitle1">Seleção</Typography>
                  </Stack>
                  <ButtonGroup variant="outlined" size="small">
                    <Button disabled={!hasApplied} onClick={toggleAll}>
                      Tudo
                    </Button>
                    <Button disabled={!hasApplied} onClick={clearSelection}>
                      Nenhum
                    </Button>
                  </ButtonGroup>
                </Stack>
                <Stack
                  direction="row"
                  justifyContent="space-around"
                  alignItems="center"
                  mt={1}
                >
                  <Box textAlign="center">
                    <Typography variant="caption" color="text.secondary">
                      Total
                    </Typography>
                    <Typography variant="body2">{total}</Typography>
                  </Box>
                  <Box textAlign="center">
                    <Typography variant="caption" color="text.secondary">
                      Filtrados
                    </Typography>
                    <Typography variant="body2">{filtered.length}</Typography>
                  </Box>
                  <Box textAlign="center">
                    <Typography variant="caption" color="text.secondary">
                      Selecionados
                    </Typography>
                    <Typography variant="body2">{selIds.size}</Typography>
                  </Box>
                </Stack>
                <Divider sx={{ my: 2 }} />
              </Box>

              {/* Criar Dataset */}
              <Box display="flex" alignItems="center">
                <NoteAddIcon fontSize="small" sx={{ mr: 1 }} />
                <Typography variant="subtitle1">Criar Dataset</Typography>
              </Box>
              <Stack spacing={1}>
                <TextField
                  size="small"
                  label="Nome do Dataset"
                  fullWidth
                  value={dsName}
                  onChange={e => setDsName(e.target.value)}
                />
                {!isMobile ? (
                  <Button
                    size="small"
                    variant="contained"
                    disabled={!dsName}
                    onClick={() => setOpenSummary(true)}
                  >
                    Criar Dataset
                  </Button>
                ) : (
                  <Fab
                    color="primary"
                    size="small"
                    onClick={() => setOpenSummary(true)}
                  >
                    <NoteAddIcon />
                  </Fab>
                )}
              </Stack>
            </Stack>
          </Paper>
        </Grid>
      </Grid>

      {/* Preview Drawer */}
      <Drawer
        anchor="right"
        open={Boolean(previewExp)}
        onClose={() => setPreviewExp(null)}
        PaperProps={{ sx: { width: { xs: '100%', sm: '70%', md: '60%' } } }}
      >
        <Box sx={{ p: 3 }}>
          <CloseIcon
            sx={{ cursor: 'pointer', mb: 2 }}
            onClick={() => setPreviewExp(null)}
          />
          {previewExp ? (
            <>
              <QuickGraficoViewer experiment={previewExp} />
              <Box textAlign="center" mt={2}>
                <Button variant="contained" onClick={() => toggle(previewExp.id)}>
                  {selIds.has(previewExp.id)
                    ? 'Remover do dataset'
                    : 'Adicionar ao dataset'}
                </Button>
              </Box>
            </>
          ) : (
            <Typography color="text.secondary">
              Nenhum experimento selecionado.
            </Typography>
          )}
        </Box>
      </Drawer>

      {/* Summary Dialog */}
      <Dialog open={openSummary} onClose={() => setOpenSummary(false)}>
        <Box sx={{ minWidth: 360 }}>
          <Typography variant="h6" p={2}>
            Resumo do Dataset
          </Typography>
          <Divider />
          <List>
            <ListItem>
              <ListItemText primary="Nome" secondary={dsName} />
            </ListItem>
            <ListItem>
              <ListItemText primary="Total Selecionado" secondary={selIds.size} />
            </ListItem>
          </List>
          <Divider />
          <Box p={2} display="flex" justifyContent="flex-end">
            <Button onClick={() => setOpenSummary(false)} sx={{ mr: 1 }}>
              Cancelar
            </Button>
            <Button
              variant="contained"
              onClick={handleConfirm}
            >
              Confirmar
            </Button>
          </Box>
        </Box>
      </Dialog>

      {/* Mobile FAB for Create */}
      {isMobile && dsName && (
        <Fab
          color="primary"
          sx={{ position: 'fixed', bottom: 24, right: 24 }}
          onClick={() => setOpenSummary(true)}
        >
          <NoteAddIcon />
        </Fab>
      )}
    </Container>
  );
}