// src/pages/DatasetPage.jsx
import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Grid,
  Card,
  CardActionArea,
  CardContent,
  Typography,
  Button,
  Stack,
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import FolderOpenIcon from '@mui/icons-material/FolderOpen';
import AddBoxIcon from '@mui/icons-material/AddBox';

export default function DatasetPage() {
  const navigate = useNavigate();

  return (
    <Box sx={{ p: { xs: 2, sm: 3, md: 4 } }}>
      {/* Header */}
      <Stack direction="row" spacing={2} alignItems="center" mb={3}>
        <Button
          startIcon={<ArrowBackIcon />}
          variant="outlined"
          size="small"
          onClick={() => navigate(-1)}
        >
          Voltar
        </Button>
        <Typography variant="h4" color="primary">
          Datasets
        </Typography>
      </Stack>

      {/* Grid de opções */}
      <Grid container spacing={4} justifyContent="center">
        {/* Card "Meus Datasets" */}
        <Grid item xs={12} sm={6} md={4}>
          <Card
            sx={{
              transition: 'transform 0.2s, box-shadow 0.2s',
              '&:hover': {
                transform: 'translateY(-4px)',
                boxShadow: 6,
              },
            }}
          >
            <CardActionArea onClick={() => navigate('/machine-learning/my-datasets')}>
              <CardContent sx={{ textAlign: 'center', py: 6 }}>
                <FolderOpenIcon sx={{ fontSize: 60 }} color="primary" />
                <Typography variant="h6" mt={2}>
                  Meus Datasets
                </Typography>
                <Typography variant="body2" color="text.secondary" mt={1}>
                  Visualize e gerencie seus datasets existentes
                </Typography>
              </CardContent>
            </CardActionArea>
          </Card>
        </Grid>

        {/* Card "Novo Dataset" */}
        <Grid item xs={12} sm={6} md={4}>
          <Card
            sx={{
              transition: 'transform 0.2s, box-shadow 0.2s',
              '&:hover': {
                transform: 'translateY(-4px)',
                boxShadow: 6,
              },
            }}
          >
            <CardActionArea onClick={() => navigate('/machine-learning/create-dataset')}>
              <CardContent sx={{ textAlign: 'center', py: 6 }}>
                <AddBoxIcon sx={{ fontSize: 60 }} color="secondary" />
                <Typography variant="h6" mt={2}>
                  Novo Dataset
                </Typography>
                <Typography variant="body2" color="text.secondary" mt={1}>
                  Crie um novo dataset selecionando experimentos
                </Typography>
              </CardContent>
            </CardActionArea>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}